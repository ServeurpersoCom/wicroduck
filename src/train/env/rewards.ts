// Reward and termination terms, ported from
// microduck_rl/src/mjlab_microduck/tasks/mdp.py and wired up the way
// microduck_standup_env_cfg.py wires them.
//
// The formulas are copied, not reinvented — including the odd-looking ones,
// because each is the residue of a run that failed some other way. Two habits
// from AGENTS.md carry over:
//
//   * Penalties here are SELF-NEGATING (they return <= 0) and take POSITIVE
//     weights. A negative weight on one of these double-negates into a reward
//     for the violation, which the policy will happily farm.
//   * Uprightness is read off the trunk quaternion as 1 - 2(qx^2 + qy^2),
//     never from a Euler angle.

import { NUM_JOINTS } from "../../sim/microduck.ts";
import type { EnvContext } from "./seams.ts";

/** Everything a term may read about the current step. */
export interface StepState {
  /** Trunk height above the floor, m. */
  z: number;
  /** Trunk quaternion [w, x, y, z]. */
  quat: Float64Array | Float32Array;
  /** Actuated joint positions, policy order. */
  jointPos: Float32Array;
  /** Target pose these joints are measured against. */
  jointTarget: Float32Array;
  /** Action this step and the one before, for smoothness terms. */
  action: Float32Array;
  prevAction: Float32Array;
}

export interface RewardTerm {
  readonly name: string;
  readonly weight: number;
  compute(ctx: EnvContext, s: StepState): number;
}

const sq = (x: number) => x * x;
/** tilt^2 in the same units the reference uses: 0 upright, 2 upside down. */
const tiltSq = (q: StepState["quat"]) => 2 * (q[1] * q[1] + q[2] * q[2]);

const gaussian = (err: number, std: number) => Math.exp(-sq(err / std));

function poseErrSq(s: StepState, indices: readonly number[]): number {
  let acc = 0;
  for (const i of indices) acc += sq(s.jointPos[i] - s.jointTarget[i]);
  return acc / indices.length;
}

/** Leg joints only — neck/head are steered by the head-pose command upstream. */
export const LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13] as const;

/** Measured trunk z at the natural standing equilibrium (HOME pose). */
export const STAND_Z = 0.115;
/** Measured seated equilibrium. */
export const SIT_Z = 0.06;

export function standupRewards(): RewardTerm[] {
  return [
    // Two-layer height Gaussian: the wide one pulls from a sit, the sharp one
    // supplies gradient in the last centimetre where the wide one has
    // saturated. Upstream converged at z = 0.109 with only the wide layer.
    {
      name: "height_stand",
      weight: 1.0,
      compute: (_c, s) => gaussian(s.z - STAND_Z, 0.04),
    },
    {
      name: "height_stand_sharp",
      weight: 1.0,
      compute: (_c, s) => gaussian(s.z - STAND_Z, 0.015),
    },
    // Self-negating: makes staying low a net cost rather than a mildly worse
    // positive, which is what stopped the policy parking in a sit.
    {
      name: "height_stand_l1",
      weight: 7.5,
      compute: (_c, s) => -Math.abs(s.z - STAND_Z),
    },
    // Linear uprightness: gradient at every tilt, including near-flat where a
    // Gaussian is numerically dead.
    {
      name: "upright_linear",
      weight: 1.5,
      compute: (_c, s) => 1 - tiltSq(s.quat),
    },
    // Sharp uprightness, but only paid at standing height — otherwise
    // "crouch low and vertical" scores as well as standing.
    {
      name: "upright_sharp",
      weight: 1.5,
      compute: (_c, s) => {
        const t = Math.min(1, Math.max(0, (s.z - SIT_Z) / Math.max(STAND_Z - SIT_Z, 1e-6)));
        const smooth = t * t * (3 - 2 * t);
        return Math.exp(-tiltSq(s.quat) / sq(0.3)) * smooth;
      },
    },
    // Always-on pose Gaussian, per joint then averaged.
    {
      name: "pose_stand_legs",
      weight: 2.0,
      compute: (_c, s) => {
        let acc = 0;
        for (const i of LEG_JOINTS) acc += gaussian(s.jointPos[i] - s.jointTarget[i], 0.5);
        return acc / LEG_JOINTS.length;
      },
    },
    {
      name: "pose_stand_l1",
      weight: 1.25,
      compute: (_c, s) => {
        let acc = 0;
        for (const i of LEG_JOINTS) acc += Math.abs(s.jointPos[i] - s.jointTarget[i]);
        return -acc / LEG_JOINTS.length;
      },
    },
    // A product, not a sum: an additive stack has a compromise basin where a
    // lean scores 80% of everything, and the policy parks there. A product
    // collapses on any single deficient factor. Stds are deliberately broad
    // so the CURRENT policy scores visibly — too tight and the gradient is
    // invisible and nothing moves.
    {
      name: "standing_composite",
      weight: 3.75,
      compute: (_c, s) =>
        gaussian(s.z - STAND_Z, 0.04) *
        Math.exp(-tiltSq(s.quat) / sq(0.4)) *
        Math.exp(-poseErrSq(s, LEG_JOINTS) / sq(0.4)),
    },
    // Smoothness. mjlab-base cost functions return >= 0 and take a NEGATIVE
    // weight — the opposite convention to the self-negating penalties above.
    {
      name: "action_rate_l2",
      weight: -0.1,
      compute: (_c, s) => {
        let acc = 0;
        for (let j = 0; j < NUM_JOINTS; j++) acc += sq(s.action[j] - s.prevAction[j]);
        return acc;
      },
    },
  ];
}

export interface TerminationTerm {
  readonly name: string;
  /** True ends the episode. */
  check(ctx: EnvContext, s: StepState): boolean;
}

export function standupTerminations(): TerminationTerm[] {
  return [
    // A solver explosion poisons every downstream statistic, so it ends the
    // episode rather than being clipped away.
    {
      name: "nan_state",
      check: (_c, s) => !Number.isFinite(s.z) || !Number.isFinite(s.quat[0]),
    },
  ];
}
