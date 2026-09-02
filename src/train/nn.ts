// A small MLP with hand-written backprop, plus Adam.
//
// Hand-written because the architecture is fixed and tiny (512/256/128,
// ~200k parameters): a framework would cost megabytes of bundle for
// generality this never needs, and the whole thing is ~200 lines. The risk of
// hand-rolling is a silently wrong gradient, which is why
// `scripts/check-grad.ts` finite-differences every parameter tensor before any
// of this is trusted.
//
// Parameters live in flat Float32Arrays so serialization, Adam and (later) a
// WebGPU upload are all just buffer copies.

/** ELU, matching the reference config's `activation="elu"`. */
const elu = (x: number) => (x >= 0 ? x : Math.exp(x) - 1);
/** d/dx ELU, expressed via the OUTPUT so the forward pass need not be kept. */
const eluGradFromOutput = (y: number) => (y >= 0 ? 1 : y + 1);

export interface Layer {
  /** Row-major [out, in]. */
  w: Float32Array;
  b: Float32Array;
  dw: Float32Array;
  db: Float32Array;
  inDim: number;
  outDim: number;
}

function makeLayer(inDim: number, outDim: number, rng: () => number, gain: number): Layer {
  const w = new Float32Array(inDim * outDim);
  // Orthogonal-ish init is what rsl_rl uses; He-scaled uniform is close enough
  // for a net this small and avoids a QR decomposition here.
  const scale = gain * Math.sqrt(2 / inDim);
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * scale;
  return {
    w,
    b: new Float32Array(outDim),
    dw: new Float32Array(inDim * outDim),
    db: new Float32Array(outDim),
    inDim,
    outDim,
  };
}

export class MlpNet {
  readonly layers: Layer[] = [];
  /** Post-activation output of each layer, for the current batch. */
  #acts: Float32Array[] = [];
  /** Gradient scratch, one per layer input. */
  #grads: Float32Array[] = [];
  #batch = 0;
  #input: Float32Array | null = null;

  constructor(
    inDim: number,
    outDim: number,
    hidden: readonly number[],
    rng: () => number = Math.random,
    /** rsl_rl scales the final layer down so the initial policy is near-zero. */
    outputGain = 0.01,
  ) {
    const dims = [inDim, ...hidden, outDim];
    for (let i = 0; i < dims.length - 1; i++) {
      const last = i === dims.length - 2;
      this.layers.push(makeLayer(dims[i], dims[i + 1], rng, last ? outputGain : 1));
    }
  }

  get paramCount(): number {
    return this.layers.reduce((n, l) => n + l.w.length + l.b.length, 0);
  }

  #ensureBatch(batch: number): void {
    if (this.#batch === batch) return;
    this.#batch = batch;
    this.#acts = this.layers.map((l) => new Float32Array(batch * l.outDim));
    this.#grads = this.layers.map((l) => new Float32Array(batch * l.inDim));
  }

  /** `x` is [batch, inDim] row-major. Returns [batch, outDim]. */
  forward(x: Float32Array, batch: number): Float32Array {
    this.#ensureBatch(batch);
    this.#input = x;
    let cur = x;
    for (let li = 0; li < this.layers.length; li++) {
      const { w, b, inDim, outDim } = this.layers[li];
      const out = this.#acts[li];
      const isLast = li === this.layers.length - 1;
      for (let n = 0; n < batch; n++) {
        const xo = n * inDim, oo = n * outDim;
        for (let o = 0; o < outDim; o++) {
          const wo = o * inDim;
          let acc = b[o];
          for (let i = 0; i < inDim; i++) acc += w[wo + i] * cur[xo + i];
          out[oo + o] = isLast ? acc : elu(acc);
        }
      }
      cur = out;
    }
    return cur;
  }

  zeroGrad(): void {
    for (const l of this.layers) {
      l.dw.fill(0);
      l.db.fill(0);
    }
  }

  /**
   * `gOut` is d(loss)/d(output), [batch, outDim]. Accumulates into dw/db and
   * returns d(loss)/d(input). Call `forward` first — the activations it cached
   * are what this differentiates.
   */
  backward(gOut: Float32Array, batch: number): Float32Array {
    if (!this.#input) throw new Error("backward() before forward()");
    let g = gOut;
    for (let li = this.layers.length - 1; li >= 0; li--) {
      const { w, dw, db, inDim, outDim } = this.layers[li];
      const input = li === 0 ? this.#input : this.#acts[li - 1];
      const gIn = this.#grads[li];
      gIn.fill(0);
      const isLast = li === this.layers.length - 1;
      const act = this.#acts[li];

      for (let n = 0; n < batch; n++) {
        const go = n * outDim, xo = n * inDim;
        for (let o = 0; o < outDim; o++) {
          // Chain through the activation before touching the weights.
          const gy = isLast ? g[go + o] : g[go + o] * eluGradFromOutput(act[go + o]);
          if (gy === 0) continue;
          db[o] += gy;
          const wo = o * inDim;
          for (let i = 0; i < inDim; i++) {
            dw[wo + i] += gy * input[xo + i];
            gIn[xo + i] += gy * w[wo + i];
          }
        }
      }
      g = gIn;
    }
    return g;
  }

  /** Every parameter tensor paired with its gradient, for Adam and checkpoints. */
  tensors(): { param: Float32Array; grad: Float32Array }[] {
    return this.layers.flatMap((l) => [
      { param: l.w, grad: l.dw },
      { param: l.b, grad: l.db },
    ]);
  }
}

/** Global L2 norm over every gradient, then scale down if it exceeds `max`. */
export function clipGradNorm(tensors: { grad: Float32Array }[], max: number): number {
  let sumSq = 0;
  for (const { grad } of tensors) for (let i = 0; i < grad.length; i++) sumSq += grad[i] * grad[i];
  const norm = Math.sqrt(sumSq);
  if (norm > max && norm > 0) {
    const scale = max / norm;
    for (const { grad } of tensors) for (let i = 0; i < grad.length; i++) grad[i] *= scale;
  }
  return norm;
}

export class Adam {
  readonly #m: Float32Array[];
  readonly #v: Float32Array[];
  #t = 0;
  lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;

  constructor(
    tensors: { param: Float32Array }[],
    lr: number,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
  ) {
    this.#m = tensors.map((t) => new Float32Array(t.param.length));
    this.#v = tensors.map((t) => new Float32Array(t.param.length));
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  step(tensors: { param: Float32Array; grad: Float32Array }[]): void {
    this.#t++;
    const bc1 = 1 - Math.pow(this.beta1, this.#t);
    const bc2 = 1 - Math.pow(this.beta2, this.#t);
    for (let k = 0; k < tensors.length; k++) {
      const { param, grad } = tensors[k];
      const m = this.#m[k], v = this.#v[k];
      for (let i = 0; i < param.length; i++) {
        const g = grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        param[i] -= (this.lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + this.eps);
      }
    }
  }

  /** Adam's moments are part of the training state — a resumed run without
   *  them takes a visible quality hit for hundreds of steps. */
  serialize(): { t: number; m: number[][]; v: number[][] } {
    return {
      t: this.#t,
      m: this.#m.map((a) => Array.from(a)),
      v: this.#v.map((a) => Array.from(a)),
    };
  }

  load(state: { t: number; m: number[][]; v: number[][] }): void {
    this.#t = state.t;
    state.m.forEach((a, i) => this.#m[i]?.set(a));
    state.v.forEach((a, i) => this.#v[i]?.set(a));
  }
}
