// ONNX policy inference in the browser, via onnxruntime-web's WASM backend.
//
// The exported checkpoints are plain MLPs with the observation normalizer
// folded in: one float32 [1, 61] input named `obs`, one [1, 14] output named
// `actions`.

// The `/wasm` entry point drops the WebGL/WebGPU backends the app never uses.
// vite.config.ts adds the `onnxruntime-web-use-extern-wasm` resolve condition
// so the .wasm stays a sidecar instead of being base64-inlined into the bundle.
import * as ort from "onnxruntime-web/wasm";
import { assetUrl } from "../asset-url.ts";

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

export class Policy {
  private readonly session: ort.InferenceSession;
  private readonly inputName: string;
  private readonly outputName: string;

  private constructor(session: ort.InferenceSession, inputName: string, outputName: string) {
    this.session = session;
    this.inputName = inputName;
    this.outputName = outputName;
  }

  static async load(url: string): Promise<Policy> {
    configureOrt();
    const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    return new Policy(session, session.inputNames[0], session.outputNames[0]);
  }

  async run(obs: Float32Array): Promise<Float32Array> {
    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: new ort.Tensor("float32", obs, [1, obs.length]),
    };
    const out = await this.session.run(feeds);
    return out[this.outputName].data as Float32Array;
  }
}
