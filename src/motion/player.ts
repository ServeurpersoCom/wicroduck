// Playing a motion in the viewport.
//
// Two modes, and the difference between them is the whole point:
//
//   preview  the joints are WRITTEN to the model and the physics is not run.
//            The duck does exactly what the file says. This is the view for
//            checking that a motion looks the way it was meant to.
//   physics  the same angles are handed to the actuators and gravity gets a
//            vote. Expect it to fall: driven open-loop this robot topples in
//            about a second whatever it is asked to do, because standing is an
//            active behaviour rather than a pose. A policy trained on the
//            motion is what makes it survivable.
//
// Both are here rather than in the Svelte layer so the mode is one argument
// instead of two code paths that drift.

import { DECIMATION, JOINT_NAMES, NUM_JOINTS } from "../sim/microduck.ts";
import type { Simulation } from "../sim/scene.ts";
import { resolveJoints } from "../train/env/seams.ts";
import { MotionRef } from "../train/env/imitation.ts";
import type { EnvContext } from "../train/env/seams.ts";
import type { Motion } from "./format.ts";

export type PlayMode = "preview" | "physics";

export class MotionPlayer {
  readonly motion: Motion;
  /** Seconds into the motion. */
  time = 0;

  readonly #sim: Simulation;
  readonly #ref: MotionRef;
  readonly #ctx: EnvContext;
  readonly #pose = new Float32Array(NUM_JOINTS);
  readonly #vel = new Float32Array(NUM_JOINTS);

  constructor(sim: Simulation, motion: Motion) {
    this.#sim = sim;
    this.motion = motion;
    this.#ref = new MotionRef(motion);
    // MotionRef wants an EnvContext, which is a training-side shape. Building
    // one here rather than duplicating its height-table logic keeps the
    // viewport and the reward looking at the same reference trajectory —
    // if the preview and the training target ever disagreed, one of them would
    // be lying and there would be no way to tell which.
    const joints = resolveJoints(sim.model, JOINT_NAMES);
    this.#ctx = {
      mujoco: sim.mujoco, model: sim.model, data: sim.data, joints, envId: 0, time: 0,
    };
    this.#ref.prepare(this.#ctx, sim.standKey);
    this.seek(0);
  }

  get duration(): number {
    return this.motion.duration;
  }

  /** 0..1 through the motion, for a scrubber. */
  get phase(): number {
    return this.#ref.sampler.phase(this.time);
  }

  /** True once a one-shot motion has run out. Loops never finish. */
  get finished(): boolean {
    return !this.motion.loop && this.time >= this.motion.duration;
  }

  /** Place the duck at `t` with no physics at all. */
  seek(t: number): void {
    this.time = t;
    const { mujoco, model, data, standKey } = this.#sim;
    const pose = this.#ref.poseAt(t);
    const vel = this.#ref.velocityAt(t);
    this.#pose.set(pose);
    this.#vel.set(vel);
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    data.xfrc_applied.fill(0);
    for (let j = 0; j < NUM_JOINTS; j++) {
      data.qpos[this.#ctx.joints.qpos[j]] = this.#pose[j];
      data.qvel[this.#ctx.joints.dof[j]] = this.#vel[j];
      data.ctrl[j] = this.#pose[j];
    }
    const height = this.#ref.heightAt(t);
    if (Number.isFinite(height)) data.qpos[2] = height;
    mujoco.mj_forward(model, data);
  }

  /** Put the duck at the motion's start, ready to be played. */
  rewind(): void {
    this.seek(0);
  }

  /** Advance one control step in the given mode. */
  step(dt: number, mode: PlayMode): void {
    const next = this.time + dt;
    if (mode === "preview") {
      this.seek(this.motion.loop ? next : Math.min(next, this.motion.duration));
      return;
    }
    this.time = next;
    // Open loop: the reference pose IS the target, with nothing correcting for
    // where the duck actually ended up.
    const pose = this.#ref.poseAt(this.time);
    const { mujoco, model, data } = this.#sim;
    for (let j = 0; j < NUM_JOINTS; j++) data.ctrl[j] = pose[j];
    for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
  }
}
