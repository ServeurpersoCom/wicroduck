// The PPO training loop over a VecEnv.
//
// Runs rollout and update in one place. That is deliberate for M2: a
// distributed arrangement (workers roll out, a central learner updates) only
// pays off once inference is fast, and M0 measured inference at ~70% of the
// step budget — so M3 optimises that first, and the split comes after.
//
// Checkpointing is here from the start rather than bolted on: a resumed run
// that silently differs from an uninterrupted one is the kind of bug that only
// shows up as "training got worse overnight".

import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import type { MjModel, Mujoco } from "../sim/mujoco.ts";
import { ActorCritic, HIDDEN } from "./ac-policy.ts";
import { Adam } from "./nn.ts";
import { DEFAULT_PPO, makeBuffer, ppoUpdate, type PpoConfig, type RolloutBuffer, type UpdateStats } from "./ppo.ts";
import { VecEnv, type EnvSpec } from "./env/vec-env.ts";
import type { Kernels } from "./kernels/index.ts";
import { STAND_Z } from "./env/rewards.ts";

export interface TrainerConfig {
  envs: number;
  /** Control steps collected per environment per iteration. */
  stepsPerIter: number;
  ppo: PpoConfig;
  hidden: readonly number[];
  seed: number;
  initStd: number;
}

export const DEFAULT_TRAINER: Omit<TrainerConfig, "seed"> = {
  envs: 64,
  // The reference uses 24; with far fewer environments a longer rollout keeps
  // the batch big enough for 4 minibatches to mean anything.
  stepsPerIter: 32,
  ppo: DEFAULT_PPO,
  hidden: HIDDEN,
  initStd: 1.0,
};

export interface IterationStats extends UpdateStats {
  iteration: number;
  /** Mean reward per control step over the rollout. */
  rewardPerStep: number;
  /** Mean undiscounted return of episodes that FINISHED this iteration. */
  episodeReturn: number;
  episodes: number;
  standingFraction: number;
  totalSteps: number;
  rolloutMs: number;
  updateMs: number;
}

/** Everything needed to resume a run byte-identically. */
export interface Checkpoint {
  version: 1;
  iteration: number;
  totalSteps: number;
  config: TrainerConfig;
  policy: unknown;
  optimizer: ReturnType<Adam["serialize"]>;
  rngState: number;
}

/**
 * Mulberry32 whose state lives in a field rather than a closure.
 *
 * That matters for resume: VecEnv and ActorCritic capture the generator
 * FUNCTION at construction, so swapping in a fresh closure on restore would
 * leave them drawing from the old stream. Keeping one stable function over
 * mutable state means a restore actually reaches everyone holding it.
 */
class SeededRng {
  #a: number;

  constructor(seed: number) {
    this.#a = seed >>> 0;
  }

  readonly next = (): number => {
    this.#a = (this.#a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(this.#a ^ (this.#a >>> 15), 1 | this.#a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  get state(): number {
    return this.#a;
  }

  set state(value: number) {
    this.#a = value >>> 0;
  }
}

export class Trainer {
  readonly ac: ActorCritic;
  readonly env: VecEnv;
  readonly config: TrainerConfig;

  #opt: Adam;
  #buf: RolloutBuffer;
  readonly #rng: SeededRng;
  #iteration = 0;
  #totalSteps = 0;
  /** Undiscounted return accumulating per environment, reset on episode end. */
  #episodeReturn: Float32Array;

  constructor(opts: {
    mujoco: Mujoco;
    model: MjModel;
    standKey: number;
    spec: EnvSpec;
    config: TrainerConfig;
    /** SIMD kernels for the nets; null falls back to the JS path. */
    kernels?: Kernels | null;
  }) {
    this.config = opts.config;
    this.#rng = new SeededRng(opts.config.seed);
    this.env = new VecEnv({
      mujoco: opts.mujoco,
      model: opts.model,
      standKey: opts.standKey,
      spec: opts.spec,
      count: opts.config.envs,
      rng: this.#rng.next,
    });
    this.ac = new ActorCritic(
      OBS_SIZE,
      NUM_JOINTS,
      this.#rng.next,
      opts.config.initStd,
      opts.config.hidden,
      opts.kernels ?? null,
    );
    this.#opt = this.ac.makeOptimizer(opts.config.ppo.lr);
    this.#buf = makeBuffer(
      opts.config.stepsPerIter,
      opts.config.envs,
      this.ac.obsDim,
      this.ac.actDim,
    );
    this.#episodeReturn = new Float32Array(opts.config.envs);
  }

  get iteration(): number {
    return this.#iteration;
  }

  get totalSteps(): number {
    return this.#totalSteps;
  }

  get usesSimd(): boolean {
    return this.ac.actor.usesSimd;
  }

  /** One rollout + one PPO update. */
  iterate(): IterationStats {
    const { envs, stepsPerIter } = this.config;
    const buf = this.#buf;
    const obsDim = this.ac.obsDim, actDim = this.ac.actDim;

    let rewardSum = 0;
    let standingSteps = 0;
    let finishedEpisodes = 0;
    let finishedReturn = 0;
    this.env.resetBreakdown();

    const rolloutStart = performance.now();
    for (let t = 0; t < stepsPerIter; t++) {
      const obs = this.env.observations;
      buf.obs.set(obs, t * envs * obsDim);
      const { actions, logProbs, values } = this.ac.act(obs, envs, this.#rng.next);
      buf.actions.set(actions, t * envs * actDim);
      buf.logProbs.set(logProbs, t * envs);
      buf.values.set(values, t * envs);

      const { reward, done, timeout } = this.env.step(actions);
      buf.rewards.set(reward, t * envs);
      buf.dones.set(done, t * envs);
      buf.timeouts.set(timeout, t * envs);

      const z = this.env.trunkZ, upright = this.env.uprightness;
      for (let e = 0; e < envs; e++) {
        rewardSum += reward[e];
        this.#episodeReturn[e] += reward[e];
        if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) standingSteps++;
        if (done[e]) {
          finishedEpisodes++;
          finishedReturn += this.#episodeReturn[e];
          this.#episodeReturn[e] = 0;
        }
      }
    }
    buf.lastValues.set(this.ac.value(this.env.observations, envs).subarray(0, envs));
    const rolloutMs = performance.now() - rolloutStart;

    const updateStart = performance.now();
    const stats = ppoUpdate(this.ac, this.#opt, buf, this.config.ppo, this.#rng.next);
    const updateMs = performance.now() - updateStart;

    // Normalizer statistics update AFTER the update, never between rollout and
    // update. PPO's importance ratio assumes the stored log-probs came from the
    // same function the first minibatch evaluates; re-scaling the inputs in
    // between breaks that, and the resulting phantom KL drives the adaptive
    // learning rate to its floor and freezes the run.
    this.ac.normalizer.update(buf.obs, stepsPerIter * envs);

    this.#iteration++;
    const total = stepsPerIter * envs;
    this.#totalSteps += total;

    return {
      ...stats,
      iteration: this.#iteration,
      rewardPerStep: rewardSum / total,
      episodeReturn: finishedEpisodes > 0 ? finishedReturn / finishedEpisodes : 0,
      episodes: finishedEpisodes,
      standingFraction: standingSteps / total,
      totalSteps: this.#totalSteps,
      rolloutMs,
      updateMs,
    };
  }

  /**
   * Deterministic-policy evaluation: no sampling noise, which is what a
   * deployed policy actually does.
   *
   * Resets every environment first, ALIGNED, so each is measured over the same
   * window — without that the measurement starts from wherever the last
   * rollout left things, which makes two evaluations incomparable and made a
   * checkpoint round-trip look broken when it was only being measured from a
   * different state.
   *
   * Training resumes from a re-staggered set afterwards: leaving the pool
   * phase-locked would poison every rollout that follows.
   */
  evaluate(steps: number): { rewardPerStep: number; standingFraction: number } {
    const envs = this.config.envs;
    this.env.resetAll({ stagger: false });
    let rewardSum = 0, standingSteps = 0;
    for (let t = 0; t < steps; t++) {
      const mean = this.ac.actMean(this.env.observations, envs);
      const { reward } = this.env.step(mean);
      const z = this.env.trunkZ, upright = this.env.uprightness;
      for (let e = 0; e < envs; e++) {
        rewardSum += reward[e];
        if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) standingSteps++;
      }
    }
    const total = steps * envs;
    this.env.resetAll();
    return { rewardPerStep: rewardSum / total, standingFraction: standingSteps / total };
  }

  checkpoint(): Checkpoint {
    return {
      version: 1,
      iteration: this.#iteration,
      totalSteps: this.#totalSteps,
      config: this.config,
      policy: this.ac.serialize(),
      optimizer: this.#opt.serialize(),
      rngState: this.#rng.state,
    };
  }

  /**
   * Restore a run. Adam's moments and the normalizer statistics come back too:
   * without them a resumed run takes a visible quality dip for hundreds of
   * steps, which is easy to misread as the task getting harder.
   */
  restore(ckpt: Checkpoint): void {
    if (ckpt.version !== 1) throw new Error(`unsupported checkpoint version ${ckpt.version}`);
    this.ac.load(ckpt.policy as ReturnType<ActorCritic["serialize"]>);
    this.#opt.load(ckpt.optimizer);
    this.#iteration = ckpt.iteration;
    this.#totalSteps = ckpt.totalSteps;
    this.#rng.state = ckpt.rngState;
  }
}
