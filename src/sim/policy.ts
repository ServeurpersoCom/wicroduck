// Policies the viewport can run.
//
// Two sources, one interface: the shipped ONNX checkpoints, and networks
// trained in the Train workspace. The controller does not care which — it
// hands over a 61-float observation and gets 14 joint targets back.

import * as ort from "onnxruntime-web/wasm";
import { assetUrl } from "../asset-url.ts";
import { NUM_JOINTS, OBS_SIZE } from "./microduck.ts";
import { ActorCritic } from "../train/ac-policy.ts";

export interface PolicyRunner {
  /** Human-readable source, for the UI. */
  readonly label: string;
  /** One observation in, one action out. Batch of 1: this is deployment. */
  run(obs: Float32Array): Promise<Float32Array>;
}

// Configured on first load, never at module scope — see loadMujoco() for why.
// Static hosting sends no COOP/COEP headers, so SharedArrayBuffer — and with
// it multi-threaded ORT — is unavailable. One thread is plenty: the net is
// ~800 KB and runs at 50 Hz.
let configured = false;
function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.wasmPaths = assetUrl("vendor/ort/");
  ort.env.wasm.numThreads = 1;
  configured = true;
}

/**
 * A shipped ONNX checkpoint.
 *
 * These were traced for single-robot deployment, so the batch dimension is
 * fixed at 1 — fine here, and the reason a vectorized replay has to call them
 * once per environment.
 */
export class Policy implements PolicyRunner {
  private readonly session: ort.InferenceSession;
  private readonly inputName: string;
  private readonly outputName: string;
  readonly label: string;

  private constructor(
    session: ort.InferenceSession,
    inputName: string,
    outputName: string,
    label: string,
  ) {
    this.session = session;
    this.inputName = inputName;
    this.outputName = outputName;
    this.label = label;
  }

  static async load(url: string, label = url.split("/").pop() ?? url): Promise<Policy> {
    configureOrt();
    const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    return new Policy(session, session.inputNames[0], session.outputNames[0], label);
  }

  async run(obs: Float32Array): Promise<Float32Array> {
    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: new ort.Tensor("float32", obs, [1, obs.length]),
    };
    const out = await this.session.run(feeds);
    return out[this.outputName].data as Float32Array;
  }
}

/**
 * A network trained in this app, restored from a checkpoint.
 *
 * Runs the distribution MEAN, not a sample: exploration noise belongs in
 * training, and a deployed policy is deterministic. The observation normalizer
 * comes back with the weights — a policy fed unnormalized observations does
 * not error, it just behaves badly, which is the whole reason the reference
 * project insists the normalizer be baked into an export.
 */
export class CheckpointPolicy implements PolicyRunner {
  readonly label: string;
  readonly iteration: number;
  #ac: ActorCritic;
  #obs = new Float32Array(OBS_SIZE);

  constructor(checkpoint: unknown, label: string) {
    const ckpt = checkpoint as {
      iteration: number;
      config?: { hidden?: number[] };
      policy: { hidden?: number[]; obsDim?: number; actDim?: number };
    };
    const hidden = ckpt.policy.hidden ?? ckpt.config?.hidden;
    if (!hidden) throw new Error("checkpoint does not record its network shape");
    this.#ac = new ActorCritic(
      ckpt.policy.obsDim ?? OBS_SIZE,
      ckpt.policy.actDim ?? NUM_JOINTS,
      Math.random,
      1,
      hidden,
      null, // the viewport runs one observation at a time; SIMD buys nothing
    );
    this.#ac.load(ckpt.policy as Parameters<ActorCritic["load"]>[0]);
    this.iteration = ckpt.iteration ?? 0;
    this.label = label;
  }

  run(obs: Float32Array): Promise<Float32Array> {
    this.#obs.set(obs);
    const mean = this.#ac.actMean(this.#obs, 1);
    // Copy: the net's output buffer is reused on the next call.
    return Promise.resolve(mean.slice(0, this.#ac.actDim));
  }
}
