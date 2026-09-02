// Messages between the main thread and the training worker.

import type { IterationStats, TrainerConfig } from "./trainer.ts";

export interface TrainInit {
  robotXml: string;
  /** "hold_pose" is the bootstrap task; "standup" is the real one. */
  task: "hold_pose" | "standup";
  config: TrainerConfig;
  /** Write a checkpoint every N iterations; 0 disables autosave. */
  autosaveEvery: number;
  checkpointName: string;
  /** Continue from the stored checkpoint of that name, if one exists. */
  resume: boolean;
}

export type ToTrainWorker =
  | { type: "init"; baseUrl: string; init: TrainInit }
  | { type: "start"; iterations: number }
  | { type: "stop" }
  | { type: "save" }
  | { type: "dispose" };

export type FromTrainWorker =
  | { type: "ready"; resumedAt: number; params: number }
  | { type: "stats"; stats: IterationStats }
  | { type: "saved"; name: string; iteration: number }
  | { type: "stopped"; iteration: number }
  | { type: "error"; message: string };
