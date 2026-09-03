// Owns the whole runtime: boots the simulation and the policy, drives the
// fixed-timestep loop, and exposes what the UI needs as reactive state.
//
// Everything under src/sim/ and src/render/ stays framework-agnostic — this is
// the one file that knows about Svelte, so the sim can be driven from a test,
// a worker or a different shell without dragging the UI along.

import { loadSimulation, type Simulation } from "../sim/scene.ts";
import { CheckpointPolicy, Policy, type PolicyRunner } from "../sim/policy.ts";
import { listCheckpoints, loadCheckpoint } from "../train/checkpoint-store.ts";
import { MicroduckController, type Phase } from "../sim/controller.ts";
import { CTRL_DT, TRUNK_BODY } from "../sim/microduck.ts";
import { Viewer } from "../render/viewer.ts";
import { assetUrl } from "../asset-url.ts";
import { MotionPlayer, type PlayMode } from "../motion/player.ts";
import { listMotions, loadMotion, type MotionEntry } from "../motion/motion-store.ts";

export const POLICY_NAME = "alpha_stand.onnx";
const POLICY_URL = assetUrl(`policies/${POLICY_NAME}`);

/** Where a policy came from. `checkpoint` entries are runs trained here. */
export interface PolicyOption {
  id: string;
  label: string;
  kind: "shipped" | "checkpoint";
}

/** Never advance more than this much sim time per frame: after a tab switch
 *  the elapsed time can be seconds, and catching up would freeze the page. */
const MAX_CATCHUP_S = 0.25;

/** Pause before the next knock-down when auto-repeat is on. */
const REPEAT_DELAY_MS = 1200;

export const PHASE_LABEL: Record<Phase, string> = {
  standing: "Standing",
  limp: "Falling",
  settling: "Settling",
  recovering: "Getting up",
};

export class Session {
  /** Boot progress, for the loading overlay. */
  loadStage = $state("Starting…");
  loadProgress = $state(0);
  ready = $state(false);
  /** True once the viewport has been opened at least once. Until then no model
   *  has been downloaded — landing on the Guide should not pull 21 MB. */
  started = $state(false);
  error = $state<string | null>(null);

  /** Live telemetry, as separate primitives: assigning an unchanged value is a
   *  no-op in Svelte, so a 60 Hz update loop only repaints what moved. */
  phase = $state<Phase>("standing");
  /** 0 = flat on the floor, 100 = perfectly upright. */
  uprightPct = $state(0);
  heightCm = $state(0);
  policyHz = $state(0);

  /** Everything the viewport can be driven by: the shipped checkpoint plus
   *  every run saved from the Train workspace. */
  policies = $state<PolicyOption[]>([{ id: POLICY_NAME, label: POLICY_NAME, kind: "shipped" }]);
  activePolicy = $state(POLICY_NAME);
  policyError = $state<string | null>(null);
  policyBusy = $state(false);

  /** Motion playback. Null means the policy is driving, as usual. */
  motions = $state<MotionEntry[]>([]);
  activeMotion = $state<string | null>(null);
  motionMode = $state<PlayMode>("preview");
  motionPlaying = $state(false);
  /** 0..1 through the motion, for the scrubber. */
  motionPhase = $state(0);
  motionError = $state<string | null>(null);
  /** Mirrors of the player's identity. Reactive state rather than getters over
   *  #player: a plain private field is invisible to Svelte, so the status bar
   *  would never notice a motion being selected. */
  motionName = $state<string | null>(null);
  motionDuration = $state(0);
  #player: MotionPlayer | null = null;

  #autoRepeat = $state(false);
  get autoRepeat(): boolean {
    return this.#autoRepeat;
  }

  #showCollision = $state(false);
  /** Label for whichever policy is currently driving. */
  get activePolicyLabel(): string {
    return this.policies.find((p) => p.id === this.activePolicy)?.label ?? this.activePolicy;
  }

  get showCollision(): boolean {
    return this.#showCollision;
  }
  set showCollision(value: boolean) {
    this.#showCollision = value;
    this.#viewer?.setCollisionVisible(value);
  }

  /**
   * False while another workspace is up front. The loop then does nothing at
   * all — not just skipping the draw. Stepping on in the background would burn
   * a core and, worse, quietly bias the throughput harness measuring on the
   * other tab-half.
   */
  #active = true;

  #viewer: Viewer | null = null;
  #sim: Simulation | null = null;
  #controller: MicroduckController | null = null;
  #frame = 0;
  /** Set when the loop is resumed, so the first tick after it starts fresh. */
  #resumeAt = 0;
  #repeatTimer: ReturnType<typeof setTimeout> | undefined;
  #disposed = false;

  async start(canvas: HTMLCanvasElement): Promise<void> {
    this.started = true;
    try {
      const viewer = new Viewer(canvas);
      this.#viewer = viewer;
      // Render the (empty) scene straight away so the loading overlay sits
      // over something rather than a canvas that flashes at first paint.
      viewer.render();

      const sim = await loadSimulation((stage, done, total) => {
        this.loadStage = total ? `${stage} ${done}/${total}` : stage;
        this.loadProgress = total ? ((done ?? 0) / total) * 100 : 0;
      });
      if (this.#disposed) return;
      this.#sim = sim;

      this.loadStage = "Loading policy";
      this.loadProgress = 100;
      const policy = await Policy.load(POLICY_URL);
      if (this.#disposed) return;

      const trunkId = sim.mujoco.mj_name2id(
        sim.model, sim.mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY,
      );
      viewer.build(sim.model, { followBody: trunkId });
      viewer.setCollisionVisible(this.#showCollision);
      viewer.sync(sim.model, sim.data);

      // Awaited, not fired-and-forgotten: the selector must be populated
      // before `ready` flips, or a run saved earlier is missing from it until
      // something else happens to trigger a refresh.
      await this.refreshPolicies();
      await this.refreshMotions();
      const controller = new MicroduckController(sim, policy);
      controller.onRecoveryEnd = () => {
        if (!this.#autoRepeat) return;
        this.#repeatTimer = setTimeout(() => controller.knockDown(), REPEAT_DELAY_MS);
      };
      this.#controller = controller;

      this.ready = true;
      this.#run();
    } catch (err) {
      console.error(err);
      this.error = err instanceof Error ? err.message : String(err);
    }
  }

  // Physics runs on a fixed 50 Hz control clock decoupled from the display
  // refresh, so the duck behaves the same on a 60 Hz laptop and a 144 Hz
  // monitor. One control step at a time: inference is async, and overlapping
  // runs would feed the policy a stale observation.
  #run(): void {
    let last = performance.now() / 1000;
    let accumulator = 0;
    let steps = 0;
    let window = 0;

    const tick = async (): Promise<void> => {
      if (this.#disposed) return;
      const controller = this.#controller;
      const sim = this.#sim;
      const viewer = this.#viewer;
      if (!controller || !sim || !viewer) return;

      const now = performance.now() / 1000;
      if (!this.#active) {
        // Idle: keep the frame loop alive so the workspace can come back, but
        // do no work and let no time accumulate.
        last = now;
        this.#frame = requestAnimationFrame(() => void tick());
        return;
      }
      if (this.#resumeAt) {
        last = this.#resumeAt;
        this.#resumeAt = 0;
      }
      const dt = Math.min(now - last, MAX_CATCHUP_S);
      last = now;
      accumulator += dt;

      while (accumulator >= CTRL_DT) {
        accumulator -= CTRL_DT;
        const player = this.#player;
        if (player && this.motionPlaying) {
          // The motion drives; the policy is out of the loop entirely. In
          // preview mode that means no physics at all — the duck is placed,
          // not simulated.
          player.step(CTRL_DT, this.motionMode);
          this.motionPhase = player.phase;
          if (player.finished) this.motionPlaying = false;
        } else {
          await controller.step();
        }
        steps++;
      }

      window += dt;
      if (window >= 0.5) {
        this.policyHz = Math.round(steps / window);
        steps = 0;
        window = 0;
      }

      const t = controller.telemetry();
      this.phase = t.phase;
      this.uprightPct = Math.round(Math.max(0, -t.gravityZ) * 100);
      this.heightCm = Math.round(t.height * 1000) / 10;

      viewer.sync(sim.model, sim.data);
      viewer.render();
      this.#frame = requestAnimationFrame(() => void tick());
    };
    void tick();
  }

  async refreshMotions(): Promise<void> {
    this.motions = (await listMotions()).filter((m) => !m.error);
    if (this.activeMotion && !this.motions.some((m) => m.id === this.activeMotion)) {
      await this.selectMotion(null);
    }
  }

  /**
   * Choose a motion to play, or null to hand the duck back to the policy.
   *
   * Selecting one places the duck at the motion's first frame rather than
   * starting playback — the still pose is usually what you want to look at
   * first, and it makes the scrubber meaningful before anything moves.
   */
  async selectMotion(id: string | null): Promise<void> {
    this.motionPlaying = false;
    this.motionError = null;
    this.motionPhase = 0;
    if (!id) {
      this.#player = null;
      this.activeMotion = null;
      this.motionName = null;
      this.motionDuration = 0;
      this.#controller?.reset();
      return;
    }
    const sim = this.#sim;
    if (!sim) return;
    try {
      const motion = await loadMotion(id);
      if (!motion) throw new Error(`motion "${id}" is gone`);
      this.#player = new MotionPlayer(sim, motion);
      this.activeMotion = id;
      this.motionName = motion.name;
      this.motionDuration = motion.duration;
    } catch (err) {
      this.#player = null;
      this.activeMotion = null;
      this.motionName = null;
      this.motionDuration = 0;
      this.motionError = err instanceof Error ? err.message : String(err);
    }
  }

  playMotion(): void {
    if (!this.#player) return;
    // Physics playback from a half-finished preview would start mid-air with
    // the velocities the file implies; restarting is the honest thing to show.
    if (this.#player.finished || this.motionMode === "physics") this.#player.rewind();
    this.motionPlaying = true;
  }

  pauseMotion(): void {
    this.motionPlaying = false;
  }

  rewindMotion(): void {
    this.#player?.rewind();
    this.motionPhase = 0;
  }

  /** Scrub to a fraction of the motion. Preview only — seeking teleports, and
   *  teleporting a physics run is not a thing that means anything. */
  seekMotion(phase: number): void {
    const player = this.#player;
    if (!player) return;
    this.motionPlaying = false;
    player.seek(phase * player.duration);
    this.motionPhase = player.phase;
  }

  /** Re-read the saved runs. Cheap, and the Train workspace can add one at
   *  any time. */
  async refreshPolicies(): Promise<void> {
    const saved = await listCheckpoints();
    this.policies = [
      { id: POLICY_NAME, label: POLICY_NAME, kind: "shipped" },
      ...saved.map((c) => ({ id: `ckpt:${c.name}`, label: c.name, kind: "checkpoint" as const })),
    ];
    // A run deleted elsewhere should not leave the selector pointing at it.
    if (!this.policies.some((p) => p.id === this.activePolicy)) {
      this.activePolicy = POLICY_NAME;
    }
  }

  /** Swap the driving policy in place — the duck keeps its current pose. */
  async selectPolicy(id: string): Promise<void> {
    if (!this.#controller || this.policyBusy) return;
    this.policyBusy = true;
    this.policyError = null;
    try {
      let runner: PolicyRunner;
      if (id.startsWith("ckpt:")) {
        const name = id.slice(5);
        const ckpt = await loadCheckpoint<unknown>(name);
        if (!ckpt) throw new Error(`checkpoint "${name}" is gone`);
        runner = new CheckpointPolicy(ckpt, name);
      } else {
        runner = await Policy.load(POLICY_URL, POLICY_NAME);
      }
      this.#controller.setPolicy(runner);
      this.activePolicy = id;
    } catch (err) {
      this.policyError = err instanceof Error ? err.message : String(err);
    } finally {
      this.policyBusy = false;
    }
  }

  setActive(value: boolean): void {
    this.#active = value;
    // Resuming after a pause must not replay the elapsed wall time as sim time.
    if (value) this.#resumeAt = performance.now() / 1000;
  }

  knockDown(): void {
    this.#controller?.knockDown();
  }

  push(): void {
    this.#controller?.push();
  }

  standUp(): void {
    this.#controller?.standUp();
  }

  reset(): void {
    this.setAutoRepeat(false);
    this.#controller?.reset();
  }

  /** Switching it on knocks the duck over straight away; switching it off
   *  cancels a queued repeat but leaves the current attempt alone. */
  setAutoRepeat(value: boolean): void {
    this.#autoRepeat = value;
    if (value) this.knockDown();
    else clearTimeout(this.#repeatTimer);
  }

  destroy(): void {
    this.#disposed = true;
    cancelAnimationFrame(this.#frame);
    clearTimeout(this.#repeatTimer);
    this.#viewer?.dispose();
    this.#viewer = null;
  }
}
