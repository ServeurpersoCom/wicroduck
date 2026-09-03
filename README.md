# Wicroduck

An all-in-one, **100% in-browser** toolchain for [Microduck](https://github.com/pollen-robotics/microduck) —
the ~800 g, ~25 cm bipedal robot duck. No CUDA box, no Python install, no
backend: open a tab and the whole loop runs there.

The reference training stack lives in [`microduck_rl/`](microduck_rl) (a
submodule of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)):
mjlab + MuJoCo Warp + PPO on a GPU, exported to ONNX. This repo is the attempt
to put that pipeline — simulate, run, and eventually *train* — behind a URL.

**Where it is now: the stand-up demo.** A Svelte + Vite app that runs MuJoCo
compiled to WebAssembly, steps the real Microduck MJCF at 200 Hz, and drives it
with the shipped `alpha_stand` ONNX policy at 50 Hz. Knock the duck over and
watch it get itself back on its feet, rendered from the compiled model's own
geometry.

## Run it

```bash
git submodule update --init --recursive
npm install
npm run dev          # http://localhost:5173
```

`npm run dev` and `npm run build` both run `scripts/prepare-assets.mjs` first,
which populates `public/` (all gitignored):

| Path              | Contents                                                        | Source |
| ----------------- | --------------------------------------------------------------- | ------ |
| `public/vendor/`  | `mujoco.js` + `mujoco.wasm`, onnxruntime's wasm sidecars          | `node_modules/` |
| `public/model/`   | `robot_allcollisions.xml` + the 38 STL meshes it references       | the `microduck_rl` submodule |
| `public/policies/`| `alpha_stand.onnx`                                                | [`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies) |

## The app

A studio-style shell: a left rail switches workspaces, an inspector on the
right holds the controls for the active one, and a status bar carries live
telemetry.

- **Guide** — the landing page: what this does, and a worked example of
  teaching the duck a new skill.
- **Simulate** — the viewport and the stand-up demo.
- **Train** — the throughput harness, the vectorized environment, and PPO.

The simulator loads on the first visit to Simulate rather than at startup —
it is a 21 MB download and nobody reading the Guide asked for it. After that
the stage stays mounted across workspace switches, because booting MuJoCo and
the policy takes seconds; it is fully paused while hidden, so a duck stepping
in the background cannot skew the throughput harness next door.

| Control | What it does |
| ------- | ------------ |
| **Knock down & stand up** | Drops the duck in a random tumbled pose, limp. The state machine takes it from there. |
| **Push** | Shoves the trunk from a random direction for 60 ms with the policy still driving — a light shove gets absorbed, a hard one topples it and the fall detection kicks in. |
| **Stand up** | Starts a get-up attempt from wherever the duck is now. |
| **Reset** | Back to the STAND keyframe. |
| **Auto-repeat** | Knocks the duck over again a second after each successful get-up. |
| **Collision geoms** | Overlays the collision proxies as wireframe. |

Drag to orbit, scroll to zoom; the camera follows the trunk.

## Checking it without a browser

```bash
npm run check           # everything below, in order
npm run check:standup   # the deployment loop: does the duck get up?
npm run check:env       # the training env: does the reward stack rank behaviour?
npm run check:grad      # finite-difference the hand-written backprop
npm run check:ppo       # can PPO solve a toy task at all?
npm run check:kernels   # do the SIMD kernels match the JS reference?
npm run check:trainer   # can it learn on the robot, and does a checkpoint restore?
npm run typecheck
```

`check:standup` runs the same physics and the same policy under Node and
asserts the duck stands up from five tumbled poses. It re-derives the
observation vector independently of `src/`, so a mismatch between the two is
exactly what it catches.

`check:env` replays three controllers — the shipped `alpha_stand`, a
do-nothing baseline and random noise — through the training environment with
the same seed, and asserts the reward stack ranks them correctly. A reward
function cannot be validated alone: "alpha_stand scores 8.9" means nothing
until you know that doing nothing scores 1.6. Unlike `check:standup` this one
imports the real `src/` modules (Node 24 strips the types), so it tests the
code the trainer will run.

`check:grad`, `check:ppo` and `check:trainer` gate the learner in that order —
each one is cheap and the next is only meaningful if it passes. Hand-rolled
gradients fail *silently*: the loss still goes down, just to the wrong place,
and you find out an hour into a run.

All of them are far faster to iterate on than a browser, which is why the
training work leans on them.

## How the loop works

The policy is blind: it sees one 61-float vector and nothing else. Getting that
vector right *is* the port, and every constant in `src/sim/microduck.ts` is
copied from `microduck_rl/scripts/infer_policy.py` and the ONNX metadata.

```
obs = [ base_ang_vel(3) | projected_gravity(3) | joint_pos(14)
      | joint_vel(14)   | last_action(14)      | command(13) ]        → 61

ctrl[j] = DEFAULT_POSE[j] + action[j] * ACTION_SCALE                  → 14
```

- `joint_pos` is *relative to* `DEFAULT_POSE` (the STAND2 reference pose), and
  actions are offsets from the same pose — it is the zero of both spaces.
- `projected_gravity` is world `-Z` rotated into the trunk frame. Its z
  component is the uprightness signal the whole state machine keys off: `-1`
  standing, `0` on its side.
- `command` is all zeros for the get-up policy. Other checkpoints put a twist,
  a head pose or a phase encoding in those slots.
- Physics steps at 200 Hz, the policy at 50 Hz (`DECIMATION = 4`). Both rates
  are training parameters, not performance knobs.

`src/sim/controller.ts` wraps that in the fall-recovery state machine the real
robot's runtime uses:

```
standing ──(trunk tipped past -0.5 for 0.3 s)──> settling ──(0.3 s)──> recovering
   ^                                                                       │
   └────────────────(upright past -0.85 for a full second)─────────────────┘
```

`standing` and `recovering` run the same control — `alpha_stand` is a get-up
policy *and* a balance controller, and it holds the pose indefinitely once up
(`check-standup.mjs` asserts that). They are separate states only so the demo
can tell "made it" from "still trying". A deliberate knock-down inserts a
`limp` state first, so the fall itself is pure physics.

## Layout

```
src/sim/mujoco.ts          MuJoCo WASM loader + typings for the bits used
src/sim/scene.ts           MJCF assembly (floor, timestep, STAND keyframe) + compile
src/sim/microduck.ts       robot + policy interface constants
src/sim/policy.ts          ONNX inference
src/sim/controller.ts      observation -> policy -> ctrl, and the state machine
src/render/viewer.ts       generic MuJoCo-geoms -> three.js renderer

src/lib/session.svelte.ts  boots the above, owns the fixed-timestep loop,
                           exposes it as reactive state — the Svelte boundary
src/lib/views.ts           the workspace list the rail renders
src/lib/Rail.svelte        left nav
src/lib/Stage.svelte       viewport canvas + loading/error overlays
src/lib/Inspector.svelte   right panel: policy info, actions, options
src/lib/StatusBar.svelte   live telemetry strip
src/lib/TrainView.svelte   placeholder workspace
src/App.svelte             shell layout (grid) + workspace switching
src/style.css              design tokens + reset
src/main.ts                mounts App

scripts/prepare-assets.mjs vendors the runtimes, model and policies into public/
scripts/check-standup.mjs  headless policy-loop check (no browser needed)
microduck_rl/              submodule: the reference GPU training stack
```

Two deliberate lines in that list:

- `viewer.ts` builds itself from the compiled model's geom arrays and knows
  nothing about the duck, so it renders any MJCF `scene.ts` hands it.
- `session.svelte.ts` is the only file with a `$state` in it. The sim can be
  driven from a test, a worker or a different shell without dragging the UI
  along — which is what `scripts/check-standup.mjs` already does.

## Where it is going

Training is being built in milestones — see [`docs/training-plan.md`](docs/training-plan.md)
for the measured budget, the architecture and the sim2real seams.

1. ~~In-browser simulation + inference of a trained policy~~ ✅
2. ~~M0: throughput harness~~ ✅ — ~58k control steps/s on an 18-thread laptop
3. ~~M1: vectorized environment + the sim2real seams~~ ✅
4. ~~M2: PPO on CPU, checkpoint/resume~~ ✅
5. ~~M3: fast MLP kernels~~ ✅ — 7.5× per iteration; physics is now the bottleneck
6. M4: fine-tune a shipped checkpoint; M5: from-scratch, export to ONNX

## Known rough edges

- `public/model/` is ~22 MB of STL. MuJoCo needs the full meshes (the collision
  geoms *are* the meshes), but the visual copies could be decimated hard before
  this is worth deploying.
- Single-threaded ONNX only: static hosting sends no COOP/COEP headers, so
  `SharedArrayBuffer` is unavailable.

## Prior art

The stand-up demo is a from-scratch reimplementation, but the approach is not
novel — Pollen ship an official in-browser sandbox built the same way
(MuJoCo WASM + onnxruntime-web), and it is worth playing with:

- [pollen-robotics/microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) — the official sandbox (nine policies, multiplayer ghosts)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) — the shipped ONNX checkpoints
- [awesome-microduck](https://github.com/joeynyc/awesome-microduck) — community simulators, policies and tools
