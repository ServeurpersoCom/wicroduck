// Owns the whole runtime: boots the simulation and the policy, drives the
// fixed-timestep loop, and exposes what the UI needs as reactive state.
//
// Everything under src/sim/ and src/render/ stays framework-agnostic — this is
// the one file that knows about Svelte, so the sim can be driven from a test,
// a worker or a different shell without dragging the UI along.

import { loadSimulation, type Simulation } from "../sim/scene";
import { Policy } from "../sim/policy";
import { MicroduckController, type Phase } from "../sim/controller";
import { CTRL_DT, TRUNK_BODY } from "../sim/microduck";
import { Viewer } from "../render/viewer";
import { assetUrl } from "../asset-url";

export const POLICY_NAME = "alpha_stand.onnx";
const POLICY_URL = assetUrl(`policies/${POLICY_NAME}`);

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
  error = $state<string | null>(null);

  /** Live telemetry, as separate primitives: assigning an unchanged value is a
   *  no-op in Svelte, so a 60 Hz update loop only repaints what moved. */
  phase = $state<Phase>("standing");
  /** 0 = flat on the floor, 100 = perfectly upright. */
  uprightPct = $state(0);
  heightCm = $state(0);
  policyHz = $state(0);

  #autoRepeat = $state(false);
  get autoRepeat(): boolean {
    return this.#autoRepeat;
  }

  #showCollision = $state(false);
  get showCollision(): boolean {
    return this.#showCollision;
  }
  set showCollision(value: boolean) {
    this.#showCollision = value;
    this.#viewer?.setCollisionVisible(value);
  }

  /** False while another workspace is up front: the sim keeps stepping, but
   *  there is no point drawing frames nobody can see. */
  #rendering = true;

  #viewer: Viewer | null = null;
  #sim: Simulation | null = null;
  #controller: MicroduckController | null = null;
  #frame = 0;
  #repeatTimer: ReturnType<typeof setTimeout> | undefined;
  #disposed = false;

  async start(canvas: HTMLCanvasElement): Promise<void> {
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
      const dt = Math.min(now - last, MAX_CATCHUP_S);
      last = now;
      accumulator += dt;

      while (accumulator >= CTRL_DT) {
        accumulator -= CTRL_DT;
        await controller.step();
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

      if (this.#rendering) {
        viewer.sync(sim.model, sim.data);
        viewer.render();
      }
      this.#frame = requestAnimationFrame(() => void tick());
    };
    void tick();
  }

  setRendering(value: boolean): void {
    this.#rendering = value;
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
