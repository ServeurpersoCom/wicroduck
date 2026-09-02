// Reactive wrapper around the training worker.

import { assetBase } from "../asset-url.ts";
import { DEFAULT_PPO } from "../train/ppo.ts";
import { DEFAULT_TRAINER, type IterationStats, type TrainerConfig } from "../train/trainer.ts";
import type { FromTrainWorker, ToTrainWorker } from "../train/train-protocol.ts";
import { listCheckpoints, opfsAvailable, type CheckpointInfo } from "../train/checkpoint-store.ts";

const CHECKPOINT = "current";
/** Points kept for the sparkline; older ones are dropped. */
const HISTORY = 400;

export interface HistoryPoint {
  iteration: number;
  reward: number;
  standing: number;
}

export class TrainingSession {
  task = $state<"hold_pose" | "standup">("hold_pose");
  envs = $state(32);
  stepsPerIter = $state(24);
  iterations = $state(250);
  /** Smaller than the reference 512/256/128: scalar-JS backprop is the cost. */
  hiddenPreset = $state<"small" | "reference">("small");

  status = $state<"idle" | "loading" | "running" | "stopped" | "error">("idle");
  error = $state<string | null>(null);
  iteration = $state(0);
  totalSteps = $state(0);
  paramCount = $state(0);
  resumedAt = $state(0);
  last = $state<IterationStats | null>(null);
  history = $state<HistoryPoint[]>([]);
  checkpoints = $state<CheckpointInfo[]>([]);
  savedAt = $state(0);

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

  #config(): TrainerConfig {
    return {
      ...DEFAULT_TRAINER,
      envs: this.envs,
      stepsPerIter: this.stepsPerIter,
      hidden: this.hidden,
      seed: 7,
      // Actions are joint offsets in radians: the reference std of 1.0 is ~57
      // degrees of noise per joint per step, which destroys the behaviour
      // being learned long before PPO can reinforce it.
      initStd: 0.1,
      ppo: { ...DEFAULT_PPO, epochs: 4, entropyCoef: 0, desiredKl: 0.05 },
    };
  }

  async start(resume: boolean): Promise<void> {
    if (this.status === "running" || this.status === "loading") return;
    this.status = "loading";
    this.error = null;
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
        task: this.task,
        config: this.#config(),
        autosaveEvery: 25,
        checkpointName: CHECKPOINT,
        resume,
      },
    });
  }

  stop(): void {
    this.#send({ type: "stop" });
  }

  save(): void {
    this.#send({ type: "save" });
  }

  #send(msg: ToTrainWorker): void {
    this.#worker?.postMessage(msg);
  }

  dispose(): void {
    this.#worker?.terminate();
    this.#worker = null;
  }
}
