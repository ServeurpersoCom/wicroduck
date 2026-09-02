# Wicroduck web app

The browser half of the project: MuJoCo compiled to WebAssembly, ONNX policy
inference and a three.js view, with no server in the loop. Right now it ships
the **stand-up demo** — knock the duck over, watch the shipped `alpha_stand`
policy get it back on its feet.

Svelte 5 + Vite. The UI is a thin shell: everything under `src/sim/` and
`src/render/` is framework-agnostic, and `src/lib/session.svelte.ts` is the
only file that knows Svelte exists.

## Run it

```bash
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

So the submodule has to be checked out:

```bash
git submodule update --init --recursive
```

## The demo screen

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
node scripts/check-standup.mjs 5
```

Runs the same physics and the same policy under Node and asserts the duck
stands up from five different tumbled poses. It re-derives the observation
vector independently of `src/`, so a mismatch between the two is exactly what
it catches. This is the fastest way to tell "the policy loop is wrong" apart
from "the rendering is wrong".

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
src/lib/Stage.svelte       canvas + loading/error overlays + telemetry
src/lib/Controls.svelte    buttons and toggles
src/App.svelte             layout
src/main.ts                mounts App
```

Two deliberate lines in that list:

- `viewer.ts` builds itself from the compiled model's geom arrays and knows
  nothing about the duck, so it renders any MJCF `scene.ts` hands it.
- `session.svelte.ts` is the only file with a `$state` in it. The sim can be
  driven from a test, a worker or a different shell without dragging the UI
  along — which is what `scripts/check-standup.mjs` already does.

## Known rough edges

- `public/model/` is ~22 MB of STL. MuJoCo needs the full meshes (the collision
  geoms *are* the meshes), but the visual copies could be decimated hard before
  this is worth deploying.
- Single-threaded ONNX only: static hosting sends no COOP/COEP headers, so
  `SharedArrayBuffer` is unavailable.
