# In-browser training — plan

Status: **design only, nothing implemented.** The Train workspace is a placeholder.

Decisions taken (2026-09-02):

- **Browser-native first**, with the sim2real seams designed in from day one so
  BAM/DR is additive rather than a rewrite. See "The four seams".
- **Interactive loop first**, but checkpoint/resume is first-class from M2 so
  overnight runs work later without redesign.

## 1. The budget

Measured on an M-series Mac (6 performance cores), single thread, 32 envs
round-robin per model:

| Model | geoms | `MjData` | control steps/s / thread |
| --- | --- | --- | --- |
| `robot_walk` | 76 | 13.1 MB | **13,700** |
| `robot_groundcontact` | 82 | 13.3 MB | 10,100 |
| `robot_allcollisions` (Simulate uses this) | 141 | 13.2 MB | 5,700 |

The reference standup recipe is 4096 envs × 24 steps × 15,000 iterations ≈
**1.47 B control steps**, which takes 1–2 h on a CUDA box.

Extrapolating `robot_walk` across ~6 threads and discounting for inference and
messaging, call it **~40 k control steps/s aggregate**:

| Target | Steps | Wall clock |
| --- | --- | --- |
| Full reference run (15,000 iters) | 1.47 B | ~10 h |
| Plausible convergence (2,000 iters) | 200 M | ~80 min |
| Bootstrap / debug (500 iters) | 50 M | ~20 min |

**Browser training is ~5–10× slower than the GPU box, not ~100×.** The duck is
small enough (21 DoF, 76 geoms) that CPU MuJoCo stays competitive. This is the
finding the whole plan rests on — re-measure it (M0) before trusting it.

Two constraints that fell out of the benchmark:

- **`MjData` is ~13 MB.** wasm32 caps at 2 GB, so ~150 envs/worker is a hard
  ceiling. `<size memory="1M"/>` shrinks the arena a lot but costs ~25%
  throughput — a knob, not a free win.
- **Batch envs inside a worker.** 32 envs round-robin ran ~1.9× faster per step
  than one env in a loop. Per-worker env count is a real tuning parameter.

## 2. Architecture

Five layers, each independently testable:

1. **Vectorized env** — `N` workers × `K` envs. Each worker owns one `MjModel`,
   `K` `MjData`, and does reset / DR / obs / reward / termination locally,
   returning packed `Float32Array`s by transfer. No `SharedArrayBuffer`, so no
   COOP/COEP requirement on the host.
2. **Rollout inference** — the policy runs *inside* each worker as a `[K×61]`
   GEMM, not a per-env GEMV. Batched, and no cross-thread barrier per control
   step.
3. **Env spec** — a declarative term list (reward / termination / event /
   curriculum) mirroring mjlab's manager pattern, so terms are composable and
   unit-testable one at a time.
4. **Learner** — hand-rolled MLP + backprop + Adam + GAE + clipped surrogate.
   The architecture is fixed and tiny (512/256/128, ~197 k params); a framework
   would cost megabytes for generality we do not need. CPU first, WebGPU once
   it is the measured bottleneck.
5. **UI** — run config, live reward/loss charts, a live-rendered training env,
   checkpoint list, ONNX export landing straight in Simulate.

**The training loop must live in workers, not `requestAnimationFrame`.**
Background tabs throttle rAF to ~1 Hz, which would silently stall an overnight
run. The main thread only renders and charts.

## 3. The four seams

Browser-native today, sim2real later, without a rewrite. Each of these is an
interface from the first commit, with a trivial implementation now:

| Seam | Now (browser-native) | Later (sim2real) |
| --- | --- | --- |
| `Actuator` | `XmlPositionActuator` — write targets to `data.ctrl`, MJCF `position` does the PD (kp 0.55) | `BamActuator`: the M6 voltage model + fitted `xl330` parameters, from [Rhoban/bam](https://github.com/Rhoban/bam) |
| `DelayedActuator` | lag 0 | lag 3–6 control steps (`_BAM_ACTUATOR_KWARGS`) — cheap, so build it now |
| `Randomizer` registry | empty | CoM, head CoM, mass/inertia, armature, joint friction, encoder bias, IMU misalignment, velocity pushes |
| `ObsPipeline` | raw | + per-term noise, encoder bias, IMU misalignment |

Two rules the interfaces must enforce, both learned the hard way upstream (see
`microduck_rl/AGENTS.md`):

- **Randomizers restore-then-apply.** An accumulating CoM randomizer silently
  degraded every long run for months. Make non-accumulation the interface's
  contract, not each implementation's responsibility.
- **Joint indices resolve by name, never by position.** Backlash and roller
  models interleave `passive_*` joints into `qpos`. `src/sim/controller.ts`
  already does this; the env spec must too.

Model choice (`walk` / `groundcontact` / `allcollisions` / `*_backlash`) is
config, never hardcoded.

Sizing the sim2real port: BAM is one `compute()` (a voltage-controlled DC motor
with a Coulomb + Stribeck + load-dependent friction budget) plus a parameter
block. `friction_dr_bam.py` in the submodule is a ~112-line subclass; the model
itself lives in the `bam` package. Bounded, but not free.

## 4. Verification — before any training

RL implementations fail silently and slowly: you burn an hour of compute to
learn nothing. Each step below gates the next.

1. **Replay `alpha_stand` through our env and check the reward curve.** We
   already have a known-good policy. If our ported reward stack does not score
   it highly, the reward stack is wrong. Cheap, and nothing else gives that
   signal.
2. **Gradient-check** backprop against finite differences.
3. **Solve a toy task** (point-mass reach) end to end. If PPO cannot do that in
   seconds, it will not do standup in an hour.
4. **Bootstrap task: "hold the STAND pose"** from a noisy init. Trivially
   learnable — the hello-world that proves the full loop.
5. **Only then** the real sit→stand reward stack.

## 5. Checkpoint / resume

First-class from M2. `TrainerState` covers policy + critic parameters, Adam
moments, obs-normalizer running mean/var, iteration counter, curriculum state,
and RNG streams — anything whose loss would make a resumed run differ from an
uninterrupted one.

Persist to OPFS, survive a reload, and support download/upload so a run can
move between machines. Resume is also what makes overnight runs tolerable.

## 6. Milestones

| | Goal | Gate |
| --- | --- | --- |
| **M0** | Throughput harness in the Train view: worker-count × envs/worker sweep, live steps/s | Real numbers replace section 1's estimates |
| **M1** | Vectorized env in workers + the four seams | `alpha_stand` replay scores as expected |
| **M2** | PPO on CPU + checkpoint/resume | Toy task solved, then "hold the pose" |
| **M3** | WebGPU learner, SIMD rollout inference | Iteration time low enough to watch |
| **M4** | Fine-tune from a shipped checkpoint | A visibly adapted policy |
| **M5** | From-scratch standup, ONNX export, round-trip into Simulate | A policy we trained, running in the Simulate tab |

**M4 before M5 on purpose.** Fine-tuning converges in far fewer steps and is
the most likely thing to actually work in a tab. If M3's numbers disappoint,
M4 is the product and M5 becomes a stretch goal.

## 7. Known caveats

- **Browser-trained ≠ sim2real-ready** until the seams are filled. A policy
  trained against MJCF `position` actuators is tuned to *our* actuator model,
  not the real servo. That `alpha_stand` transfers into our sim is a nice
  robustness result, not evidence the reverse direction works.
- **MuJoCo WASM (CPU, f64) ≠ MuJoCo Warp (GPU, f32).** Fine for browser-only
  use; another reason not to promise hardware transfer.
- **The reward playbook in `AGENTS.md` encodes 15,000-iteration lessons.** At
  small scale we will rediscover reward hacking on our own; budget for a few
  rounds of whack-a-mole.
- **WebGPU availability** gates M3. Needs a CPU fallback or an explicit
  capability gate in the UI.
