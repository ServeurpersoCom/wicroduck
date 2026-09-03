/// <reference lib="webworker" />
// Runs the PPO trainer off the main thread and streams statistics back.
//
// One worker, one VecEnv batch: rollout and update in the same place. A split
// arrangement (workers roll out, a central learner updates) only pays once
// inference is fast, and M0 measured inference at ~70% of the step budget — so
// M3 fixes that first and the split comes after.

import { setAssetBase } from "../asset-url.ts";
import { compileScene, loadModelAssets } from "../sim/scene.ts";
import { holdPoseSpec, standupSpec } from "./env/standup.ts";
import { Trainer, type Checkpoint } from "./trainer.ts";
import { loadCheckpoint, saveCheckpoint } from "./checkpoint-store.ts";
import { loadKernels } from "./kernels/index.ts";
import type { FromTrainWorker, ToTrainWorker, TrainInit } from "./train-protocol.ts";

let trainer: Trainer | null = null;
let cfg: TrainInit | null = null;
let running = false;

const post = (msg: FromTrainWorker) => self.postMessage(msg);

async function init(baseUrl: string, options: TrainInit): Promise<void> {
  setAssetBase(baseUrl);
  cfg = options;
  const assets = await loadModelAssets(() => {}, [options.robotXml]);
  const { model, standKey } = compileScene(assets, { robotXml: options.robotXml });
  const spec = options.task === "hold_pose" ? holdPoseSpec() : standupSpec();
  // Vite turns this into a static asset URL. If it fails to load — an engine
  // without SIMD, a stripped deployment — the nets fall back to JavaScript
  // rather than the worker dying.
  const kernels = await fetch(new URL("./kernels/kernels.wasm", import.meta.url))
    .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((bytes) => loadKernels(bytes))
    .catch(() => null);
  trainer = new Trainer({
    mujoco: assets.mujoco, model, standKey, spec, config: options.config, kernels,
  });

  let resumedAt = 0;
  if (options.resume) {
    const ckpt = await loadCheckpoint<Checkpoint>(options.checkpointName);
    if (ckpt) {
      trainer.restore(ckpt);
      resumedAt = ckpt.iteration;
    }
  }
  post({
    type: "ready",
    resumedAt,
    params: trainer.ac.actor.paramCount + trainer.ac.critic.paramCount,
    simd: trainer.usesSimd,
  });
}

/** `name` lets a run be saved under something memorable; the autosave keeps
 *  using the session's rolling slot so the two never fight. */
async function save(name?: string): Promise<void> {
  if (!trainer || !cfg) return;
  const target = name ?? cfg.checkpointName;
  await saveCheckpoint(target, trainer.checkpoint());
  post({ type: "saved", name: target, iteration: trainer.iteration });
}

/**
 * Iterate, yielding to the event loop between iterations.
 *
 * Without the yield a `stop` message would sit unread until the whole run
 * finished — the worker's queue is only drained when the task stack empties.
 */
async function run(iterations: number): Promise<void> {
  if (!trainer || !cfg) {
    post({ type: "error", message: "trainer not initialised" });
    return;
  }
  running = true;
  for (let i = 0; i < iterations && running; i++) {
    const stats = trainer.iterate();
    post({ type: "stats", stats });
    if (cfg.autosaveEvery > 0 && stats.iteration % cfg.autosaveEvery === 0) {
      await save();
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  running = false;
  post({ type: "stopped", iteration: trainer.iteration });
}

self.onmessage = (e: MessageEvent<ToTrainWorker>) => {
  const msg = e.data;
  const fail = (err: unknown) =>
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  try {
    if (msg.type === "init") void init(msg.baseUrl, msg.init).catch(fail);
    else if (msg.type === "start") void run(msg.iterations).catch(fail);
    else if (msg.type === "stop") running = false;
    else if (msg.type === "save") void save(msg.name).catch(fail);
    else if (msg.type === "dispose") {
      running = false;
      trainer = null;
      self.close();
    }
  } catch (err) {
    fail(err);
  }
};
