// What a run is training on, in a form that survives postMessage.
//
// The learner and every rollout worker have to build the SAME environment, and
// they are in different threads, so the task cannot be an EnvSpec — it has to
// be data. For hold-pose and stand-up a tag is enough; a motion task carries
// its file, which is why the workers take a MotionFile rather than a name they
// would each have to load from storage and could disagree about.

import { parseMotion, type MotionFile } from "../../motion/format.ts";
import { motionSpec } from "./motion.ts";
import { holdPoseSpec, standupSpec } from "./standup.ts";
import type { EnvSpec } from "./vec-env.ts";

export type TaskSpec =
  | { kind: "hold_pose" }
  | { kind: "standup" }
  | { kind: "motion"; motion: MotionFile };

export interface TaskOptions {
  episodeLengthS?: number;
}

export function buildSpec(task: TaskSpec, options: TaskOptions = {}): EnvSpec {
  switch (task.kind) {
    case "hold_pose":
      return holdPoseSpec(options);
    case "standup":
      return standupSpec(options);
    case "motion":
      // Parsed here, in the thread that will use it: validation is cheap and a
      // file that somehow arrived malformed should fail with the parser's
      // message rather than as an undefined halfway through a rollout.
      return motionSpec(parseMotion(task.motion), options);
  }
}

export function taskLabel(task: TaskSpec): string {
  if (task.kind === "motion") return `motion:${task.motion.name ?? "untitled"}`;
  return task.kind;
}
