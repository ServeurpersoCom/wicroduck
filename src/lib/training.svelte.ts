// Reactive wrapper around the training worker.

import { assetBase } from "../asset-url.ts";
import { DEFAULT_PPO } from "../train/ppo.ts";
import { DEFAULT_TRAINER, type IterationStats, type TrainerConfig } from "../train/trainer.ts";
import type { FromTrainWorker, ToTrainWorker } from "../train/train-protocol.ts";
import { deleteCheckpoint, listCheckpoints, opfsAvailable, type CheckpointInfo } from "../train/checkpoint-store.ts";
import type { TaskSpec } from "../train/env/tasks.ts";
import { listMotions, loadMotionFile, type MotionEntry } from "../motion/motion-store.ts";

/** The rolling autosave slot. Named saves live alongside it. */
const AUTOSAVE = "autosave";
/** Points kept for the sparkline; older ones are dropped. */
const HISTORY = 400;

export interface HistoryPoint {
  iteration: number;
  reward: number;
  standing: number;
}

/** The task picker's value. Motions are `motion:<id>`, where the id is the
 *  motion-store id — so a built-in and a saved file of the same name stay
 *  distinct. */
export type TaskChoice = string;
export const MOTION_TASK = "motion:";

export class TrainingSession {
  task = $state<TaskChoice>("hold_pose");
  /** Everything the task picker can offer beyond the two built-in tasks. */
  motions = $state<MotionEntry[]>([]);
  /** Environments PER rollout worker. */
  envs = $state(32);
  /** 1 keeps environments in the learner's thread; more spreads the physics
   *  out, which is where the remaining speed is. */
  rolloutWorkers = $state(Math.min(4, Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) / 2))));
  stepsPerIter = $state(24);
  iterations = $state(250);
  /** Smaller than the reference 512/256/128: scalar-JS backprop is the cost. */
  hiddenPreset = $state<"small" | "reference">("small");

  status = $state<"idle" | "loading" | "running" | "stopped" | "error">("idle");
  error = $state<string | null>(null);
  iteration = $state(0);
  totalSteps = $state(0);
  paramCount = $state(0);
  totalEnvs = $state(0);
  rolloutLabel = $state("");
  resumedAt = $state(0);
  simd = $state(false);
  last = $state<IterationStats | null>(null);
  history = $state<HistoryPoint[]>([]);
  checkpoints = $state<CheckpointInfo[]>([]);
  savedAt = $state(0);
  savedName = $state<string | null>(null);
  /** Which stored run "Resume" continues from; null means the autosave. */
  resumeFrom = $state<string | null>(null);

  readonly opfs = opfsAvailable();
  #worker: Worker | null = null;

  get hidden(): readonly number[] {
    return this.hiddenPreset === "reference" ? [512, 256, 128] : [128, 64];
  }

  get bestReward(): number {
    return this.history.reduce((b, p) => Math.max(b, p.reward), -Infinity);
  }

  async refreshCheckpoints(): Promise<void> {
    this.checkpoints = await listCheckpoints();
  }

  async refreshMotions(): Promise<void> {
    // A file that will not parse is listed with its error rather than
    // silently dropped, but it cannot be trained on.
    this.motions = (await listMotions()).filter((m) => !m.error);
    if (this.task.startsWith(MOTION_TASK) && !this.motions.some((m) => m.id === this.motionId)) {
      this.task = "hold_pose";
    }
  }

  /** The selected motion's store id, or null when a plain task is selected. */
  get motionId(): string | null {
    return this.task.startsWith(MOTION_TASK) ? this.task.slice(MOTION_TASK.length) : null;
  }

  /** Human name for the current task, for save names and labels. */
  get taskLabel(): string {
    const id = this.motionId;
    if (!id) return this.task;
    return this.motions.find((m) => m.id === id)?.name ?? id;
  }

  /**
   * Resolve the picker's value into the task the workers get.
   *
   * The motion FILE travels, not its name: the learner and every rollout
   * worker have to build the same reward, and each of them looking the name up
   * in storage is a way for them to disagree.
   */
  async #task(): Promise<TaskSpec> {
    const id = this.motionId;
    if (!id) return { kind: this.task === "standup" ? "standup" : "hold_pose" };
    const motion = await loadMotionFile(id);
    if (!motion) throw new Error(`motion "${id}" is gone`);
    return { kind: "motion", motion };
  }

  #config(): TrainerConfig {
    return {
      ...DEFAULT_TRAINER,
      envs: this.envs,
      stepsPerIter: this.stepsPerIter,
      hidden: this.hidden,
      seed: 7,
      // Exploration is left to the TASK: hold-pose is destroyed by action
      // noise, stand-up cannot be discovered without it. See ExplorationHints.
      ppo: { ...DEFAULT_PPO, epochs: 4 },
    };
  }

  async start(resume: boolean): Promise<void> {
    if (this.status === "running" || this.status === "loading") return;
    this.status = "loading";
    this.error = null;
    let task: TaskSpec;
    try {
      task = await this.#task();
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
      this.status = "error";
      return;
    }
    if (!resume) {
      this.history = [];
      this.iteration = 0;
      this.totalSteps = 0;
    }

    this.#worker?.terminate();
    const worker = new Worker(new URL("../train/train-worker.ts", import.meta.url), {
      type: "module",
      name: "wicroduck-train",
    });
    this.#worker = worker;

    worker.onmessage = (e: MessageEvent<FromTrainWorker>) => {
      const msg = e.data;
      if (msg.type === "ready") {
        this.paramCount = msg.params;
        this.simd = msg.simd;
        this.totalEnvs = msg.envs;
        this.rolloutLabel = msg.rolloutLabel;
        this.resumedAt = msg.resumedAt;
        this.iteration = msg.resumedAt;
        this.status = "running";
        this.#send({ type: "start", iterations: this.iterations });
      } else if (msg.type === "stats") {
        this.last = msg.stats;
        this.iteration = msg.stats.iteration;
        this.totalSteps = msg.stats.totalSteps;
        const next = [
          ...this.history,
          {
            iteration: msg.stats.iteration,
            reward: msg.stats.rewardPerStep,
            standing: msg.stats.standingFraction,
          },
        ];
        this.history = next.length > HISTORY ? next.slice(next.length - HISTORY) : next;
      } else if (msg.type === "saved") {
        this.savedAt = msg.iteration;
        this.savedName = msg.name;
        void this.refreshCheckpoints();
      } else if (msg.type === "stopped") {
        this.status = "stopped";
      } else {
        this.error = msg.message;
        this.status = "error";
      }
    };
    worker.onerror = (e) => {
      this.error = e.message || "training worker crashed";
      this.status = "error";
    };

    this.#send({
      type: "init",
      baseUrl: assetBase(),
      init: {
        robotXml: "robot_allcollisions-nv.xml",
        rolloutWorkers: this.rolloutWorkers,
        task,
        config: this.#config(),
        autosaveEvery: 25,
        checkpointName: resume ? (this.resumeFrom ?? AUTOSAVE) : AUTOSAVE,
        resume,
      },
    });
  }

  stop(): void {
    this.#send({ type: "stop" });
  }

  /** Save the run under a name so it can be replayed in Simulate or continued
   *  later. Returns false if the user cancelled the prompt. */
  saveAs(): boolean {
    const suggested = `${this.taskLabel}-${this.iteration}`;
    const raw = globalThis.prompt?.("Save this run as:", suggested);
    if (raw === null || raw === undefined) return false;
    // Checkpoint names become filenames in OPFS, so keep them tame.
    const name = raw.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    if (!name) return false;
    this.#send({ type: "save", name });
    return true;
  }

  async remove(name: string): Promise<void> {
    await deleteCheckpoint(name);
    if (this.resumeFrom === name) this.resumeFrom = null;
    await this.refreshCheckpoints();
  }

  #send(msg: ToTrainWorker): void {
    this.#worker?.postMessage(msg);
  }

  dispose(): void {
    this.#worker?.terminate();
    this.#worker = null;
  }
}
