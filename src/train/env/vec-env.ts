// A batch of environments sharing one compiled MjModel inside one worker.
//
// Deliberately free of DOM and asset-loading concerns: it takes an already
// compiled model and does physics, observation, reward and reset. That is what
// lets scripts/check-env.ts run the real thing under Node instead of a
// lookalike — the M1 gate is only worth anything if it tests this code.

import {
  CTRL_DT, DECIMATION, DEFAULT_POSE, GYRO_SENSOR, NUM_JOINTS, OBS_SIZE, TRUNK_BODY,
} from "../../sim/microduck.ts";
import type { MjData, MjModel, Mujoco } from "../../sim/mujoco.ts";
import { XmlPositionActuator } from "./actuators.ts";
import { identityObsPipeline } from "./obs-pipeline.ts";
import { NO_RANDOMIZERS, applyRandomizers } from "./randomizers.ts";
import type { RewardTerm, StepState, TerminationTerm } from "./rewards.ts";
import type { Actuator, EnvContext, JointIndex, ObsPipeline, Randomizer } from "./seams.ts";
import { resolveJoints } from "./seams.ts";

/** How an episode starts. */
export type ResetPose = (
  ctx: EnvContext,
  rng: () => number,
  standKey: number,
) => void;

export interface EnvSpec {
  readonly name: string;
  /** Actuated joints, in policy order. Resolved by name — never by index. */
  readonly joints: readonly string[];
  readonly rewards: readonly RewardTerm[];
  readonly terminations: readonly TerminationTerm[];
  readonly reset: ResetPose;
  /** Episode length in seconds; the reference standup task uses 6 s. */
  readonly episodeLengthS: number;
  readonly actuator?: Actuator;
  readonly randomizers?: readonly Randomizer[];
  readonly obsPipeline?: ObsPipeline;
}

export interface StepResult {
  /** [count, OBS_SIZE] row-major. */
  obs: Float32Array;
  /** [count] total weighted reward this step. */
  reward: Float32Array;
  /** [count] 1 where the episode ended THIS step. */
  done: Uint8Array;
  /** [count] 1 where the episode ended by running out of time, not by failing. */
  timeout: Uint8Array;
}

/** Per-term reward totals over an episode, for diagnosing a reward stack. */
export type RewardBreakdown = Record<string, number>;

export class VecEnv {
  readonly count: number;
  readonly joints: JointIndex;

  readonly #mujoco: Mujoco;
  readonly #model: MjModel;
  readonly #datas: MjData[];
  readonly #standKey: number;
  readonly #spec: EnvSpec;
  readonly #actuator: Actuator;
  readonly #randomizers: readonly Randomizer[];
  readonly #obsPipeline: ObsPipeline;
  readonly #gyroAdr: number;
  readonly #trunkId: number;
  readonly #maxSteps: number;

  readonly #obs: Float32Array;
  readonly #reward: Float32Array;
  readonly #done: Uint8Array;
  readonly #timeout: Uint8Array;
  readonly #prevAction: Float32Array;
  readonly #stepCount: Int32Array;
  readonly #target = new Float32Array(DEFAULT_POSE);
  readonly #scratchJoints: Float32Array;
  /** Trunk height and uprightness per env, refreshed each step. Diagnostics
   *  read these instead of calling data.body() again: every embind accessor
   *  is a wasm boundary crossing, and one per env per step is not free. */
  readonly #zOut: Float32Array;
  readonly #uprightOut: Float32Array;
  #rng: () => number;

  /** Accumulated per-term reward since the last resetBreakdown(). */
  readonly breakdown: RewardBreakdown = {};

  constructor(opts: {
    mujoco: Mujoco;
    model: MjModel;
    standKey: number;
    spec: EnvSpec;
    count: number;
    rng?: () => number;
  }) {
    this.#mujoco = opts.mujoco;
    this.#model = opts.model;
    this.#standKey = opts.standKey;
    this.#spec = opts.spec;
    this.count = opts.count;
    this.#rng = opts.rng ?? Math.random;

    this.#actuator = opts.spec.actuator ?? new XmlPositionActuator();
    this.#randomizers = opts.spec.randomizers ?? NO_RANDOMIZERS;
    this.#obsPipeline = opts.spec.obsPipeline ?? identityObsPipeline;

    this.joints = resolveJoints(opts.model, opts.spec.joints);
    this.#gyroAdr = opts.model.sensor(GYRO_SENSOR).adr;
    this.#trunkId = opts.mujoco.mj_name2id(
      opts.model, opts.mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY,
    );
    this.#maxSteps = Math.round(opts.spec.episodeLengthS / CTRL_DT);

    this.#datas = Array.from({ length: opts.count }, () => new opts.mujoco.MjData(opts.model));
    this.#obs = new Float32Array(opts.count * OBS_SIZE);
    this.#reward = new Float32Array(opts.count);
    this.#done = new Uint8Array(opts.count);
    this.#timeout = new Uint8Array(opts.count);
    this.#prevAction = new Float32Array(opts.count * NUM_JOINTS);
    this.#stepCount = new Int32Array(opts.count);
    this.#scratchJoints = new Float32Array(NUM_JOINTS);
    this.#zOut = new Float32Array(opts.count);
    this.#uprightOut = new Float32Array(opts.count);
    for (const t of opts.spec.rewards) this.breakdown[t.name] = 0;

    for (let e = 0; e < opts.count; e++) this.resetEnv(e);
  }

  get maxSteps(): number {
    return this.#maxSteps;
  }

  #ctx(envId: number): EnvContext {
    return {
      mujoco: this.#mujoco,
      model: this.#model,
      data: this.#datas[envId],
      joints: this.joints,
      envId,
    };
  }

  resetEnv(envId: number): void {
    const ctx = this.#ctx(envId);
    // Restore-then-apply lives in applyRandomizers, so DR cannot compound
    // across episodes no matter what an individual randomizer does.
    applyRandomizers(this.#randomizers, ctx, this.#rng);
    this.#spec.reset(ctx, this.#rng, this.#standKey);
    this.#mujoco.mj_forward(this.#model, ctx.data);
    this.#actuator.reset(ctx);
    this.#obsPipeline.reset(ctx, this.#rng);
    this.#prevAction.fill(0, envId * NUM_JOINTS, (envId + 1) * NUM_JOINTS);
    this.#stepCount[envId] = 0;
    this.#writeObs(envId, new Float32Array(NUM_JOINTS));
  }

  resetAll(): Float32Array {
    for (let e = 0; e < this.count; e++) this.resetEnv(e);
    return this.#obs;
  }

  resetBreakdown(): void {
    for (const k of Object.keys(this.breakdown)) this.breakdown[k] = 0;
  }

  /** Projected gravity in the trunk frame: world -Z rotated into the body. */
  #projectedGravity(data: MjData, out: Float32Array, offset: number): void {
    const q = data.body(this.#trunkId).xquat; // [w, x, y, z]
    const w = q[0], x = q[1], y = q[2], z = q[3];
    out[offset + 0] = -2 * (x * z - w * y);
    out[offset + 1] = -2 * (y * z + w * x);
    out[offset + 2] = -(1 - 2 * (x * x + y * y));
  }

  /**
   * Build one environment's observation.
   *
   * Layout must match the exported policies exactly — see docs/training-plan.md.
   * The command block is all zeros: the get-up policy was trained that way, and
   * the slots stay present rather than being dropped so the 61D interface holds
   * across the whole policy family.
   */
  #writeObs(envId: number, lastAction: Float32Array): void {
    const data = this.#datas[envId];
    const { qpos, qvel, sensordata } = data;
    const base = envId * OBS_SIZE;
    let i = base;
    for (let a = 0; a < 3; a++) this.#obs[i++] = sensordata[this.#gyroAdr + a];
    this.#projectedGravity(data, this.#obs, i); i += 3;
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = qpos[this.joints.qpos[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = qvel[this.joints.dof[j]];
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = lastAction[j];
    this.#obs.fill(0, i, base + OBS_SIZE);
    this.#obsPipeline.apply(this.#ctx(envId), this.#obs, base);
  }

  /**
   * Advance every environment one control step.
   *
   * `actions` is [count, NUM_JOINTS] row-major. Environments that finish are
   * reset immediately and their observation is the FIRST of the new episode —
   * the standard vectorized-env convention, and what the PPO buffer expects.
   */
  step(actions: Float32Array): StepResult {
    const nj = NUM_JOINTS;
    for (let e = 0; e < this.count; e++) {
      const ctx = this.#ctx(e);
      const action = actions.subarray(e * nj, (e + 1) * nj);
      this.#actuator.apply(ctx, action);
      for (let s = 0; s < DECIMATION; s++) this.#mujoco.mj_step(this.#model, ctx.data);

      const state = this.#stepState(e, action);
      let total = 0;
      for (const term of this.#spec.rewards) {
        const value = term.compute(ctx, state) * term.weight;
        this.breakdown[term.name] += value;
        total += value;
      }
      this.#reward[e] = total;

      this.#stepCount[e]++;
      const failed = this.#spec.terminations.some((t) => t.check(ctx, state));
      const timedOut = this.#stepCount[e] >= this.#maxSteps;
      this.#done[e] = failed || timedOut ? 1 : 0;
      this.#timeout[e] = !failed && timedOut ? 1 : 0;

      if (this.#done[e]) {
        this.resetEnv(e);
      } else {
        this.#prevAction.set(action, e * nj);
        this.#writeObs(e, action);
      }
    }
    return { obs: this.#obs, reward: this.#reward, done: this.#done, timeout: this.#timeout };
  }

  #stepState(envId: number, action: Float32Array): StepState {
    const data = this.#datas[envId];
    for (let j = 0; j < NUM_JOINTS; j++) {
      this.#scratchJoints[j] = data.qpos[this.joints.qpos[j]];
    }
    const quat = data.body(this.#trunkId).xquat as Float64Array;
    this.#zOut[envId] = data.qpos[2];
    this.#uprightOut[envId] = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]);
    return {
      z: data.qpos[2],
      quat,
      jointPos: this.#scratchJoints,
      jointTarget: this.#target,
      action,
      prevAction: this.#prevAction.subarray(envId * NUM_JOINTS, (envId + 1) * NUM_JOINTS),
    };
  }

  /** Read-only view of an environment's MuJoCo state, for diagnostics. */
  dataAt(envId: number): MjData {
    return this.#datas[envId];
  }

  /** Trunk height per env as of the last step. */
  get trunkZ(): Float32Array {
    return this.#zOut;
  }

  /** Uprightness per env as of the last step: 1 upright, 0 on its side. */
  get uprightness(): Float32Array {
    return this.#uprightOut;
  }

  get observations(): Float32Array {
    return this.#obs;
  }
}
