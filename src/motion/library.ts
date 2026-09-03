// Motions that ship with the app.
//
// These are written as literal FILES rather than built from DEFAULT_POSE in
// code, because their second job is to be the worked examples an author — or
// an agent reading docs/motion-format.md — copies. A file that leans on a
// constant nobody outside this repo has is a bad example.
//
// The angles are not guesses. The duck faces -X, and the directions were
// measured on the compiled model:
//
//   neck_pitch  +  leans the head forward and down (a bow); - lifts it back
//   head_pitch  -  tips the beak down (looking at its feet); + looks up
//   head_yaw    +  turns to its right
//   knee        +  on the left, - on the right folds the leg (the two legs'
//                  axes are mirrored — see DEFAULT_POSE's sign pattern)
//
// The leg angles in "squat" and "bow" are fractions of the way to the seated
// pose in standup.ts, which is a measured equilibrium, so they fold the way
// the joints actually bend rather than the way they look like they should.

import type { MotionFile } from "./format.ts";

export const BUILTIN_MOTIONS: MotionFile[] = [
  {
    format: 1,
    name: "nod",
    description: "Tip the beak down and back up. Head only — the body holds still.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: ["head_pitch"],
    keyframes: [
      { t: 0.0, pose: [0.3491] },
      { t: 0.5, pose: [-0.1] },
      { t: 1.0, pose: [0.3491] },
    ],
  },
  {
    format: 1,
    name: "look_around",
    description: "Sweep the head left and right, twice as slow as a nod.",
    loop: true,
    units: "deg",
    interp: "cubic",
    joints: ["head_yaw"],
    keyframes: [
      { t: 0.0, pose: [0] },
      { t: 0.9, pose: [40] },
      { t: 1.8, pose: [0] },
      { t: 2.7, pose: [-40] },
      { t: 3.6, pose: [0] },
    ],
  },
  {
    format: 1,
    name: "squat",
    description: "Fold halfway to a sit and rise again, keeping the trunk level.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: [
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 0.9, pose: [-0.0436, -0.4329, 0.6725, 0.2265, 0.0436, 0.4329, -0.6725, -0.2265] },
      { t: 1.8, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "bow",
    description: "Dip the head forward over a slight knee bend, hold, then rise.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "neck_pitch", "head_pitch",
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.0, pose: [0.9500, -0.0500, -0.0567, -0.4404, 0.4693, 0.2944, 0.0567, 0.4404, -0.4693, -0.2944] },
      { t: 2.0, pose: [0.9500, -0.0500, -0.0567, -0.4404, 0.4693, 0.2944, 0.0567, 0.4404, -0.4693, -0.2944] },
      { t: 3.0, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
];

export function builtinMotionNames(): string[] {
  return BUILTIN_MOTIONS.map((m) => m.name ?? "untitled");
}
