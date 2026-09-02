# In-browser training — plan

Status: **M0 done** — the Train workspace runs a throughput harness. Nothing
else is implemented yet.

Decisions taken (2026-09-02):

- **Browser-native first**, with the sim2real seams designed in from day one so
  BAM/DR is additive rather than a rewrite. See "The four seams".
- **Interactive loop first**, but checkpoint/resume is first-class from M2 so
  overnight runs work later without redesign.

## 1. The budget — measured

M0 ships a sweep in the Train workspace: each cell spins up a fresh worker
pool, allocates its environments, and steps them all at once. Numbers below are
from an M-series Mac (6 P-cores + 12 E-cores, `hardwareConcurrency` 18),
`robot_walk-nv`, 16 envs per worker, 2 s windows, Simulate paused.

| Workers | With policy forward | Physics only | Heap |
| ---: | ---: | ---: | ---: |
| 1 | 5,184 | 16,014 | 312 MB |
| 2 | 10,485 | 31,818 | 623 MB |
| 4 | 18,855 | 58,718 | 1,246 MB |
| 8 | 32,193 | 102,578 | 2,493 MB |
| 16 | **58,540** | 185,610 | 4,985 MB |
| 18 | 58,392 | **193,499** | 5,608 MB |

Control steps/s. Scaling is near-linear to 16 workers and then flat — the E-cores
stop contributing. Run-to-run variance is ±15%, so treat ~58 k as the working
figure, not 58,392.

**The plan assumed 40,000; the machine does ~58,000.** So:

| Target | Steps | Wall clock |
| --- | --- | --- |
| Full reference recipe (15,000 iters) | 1.47 B | **7 h** |
| Plausible convergence (2,000 iters) | 200 M | **56 min** |
| Bootstrap / debug (500 iters) | 50 M | **14 min** |

### What M0 overturned

**Physics is not the bottleneck — inference is.** Physics alone sustains
193,499 steps/s; adding a policy-shaped forward pass drops it to 58,392, so the
naive scalar-JS MLP eats **70% of the budget**. That is ~3× of headroom sitting
in a single function, and it re-prioritises M3: the rollout's inference path
matters far more than the learner's.

At the physics-only ceiling the full reference recipe would take 2.1 h — the
CUDA box's own number. Inference will never be free, but WASM SIMD or a batched
WebGPU pass should recover a good part of that gap.

**Visual geoms cost memory, not time.** Every model now ships a `-nv` twin with
`class="visual"` geoms and the meshes only they referenced removed —
`robot_walk-nv` is 6 geoms and 4 meshes against 76 and 38. Because visual geoms
are already `contype=0 conaffinity=0`, this buys only ~7% throughput, but it
cuts per-worker heap from 746 MB to 312 MB and makes a pool start much faster.
`prepare-assets` compiles both twins and asserts `nq`/`nv`/`nu`/`nbody` and
total mass match, so a bad strip fails the build rather than the policy.

**Memory, not CPU, caps the worker count.** 18 workers × 16 envs is 5.6 GB.
`MjData` is ~14 MB regardless of model, so envs-per-worker is the expensive
axis and worker count is the cheap one.

**Bigger per-worker batches do not help.** 64 envs/worker measured slightly
*worse* than 16 (4,949 vs 5,150 per thread). The earlier Node result suggesting
otherwise was on the un-stripped model, where collision cost dominated.

**Worker count saturates at the P-core count × ~2.7.** 16 and 18 workers tie.
Defaulting the pool to `hardwareConcurrency` is fine but not free — the last
few workers cost memory and return nothing.

**Model choice barely moves throughput.** At 8 workers: `walk-nv` 31.6k,
`groundcontact-nv` 28.0k, `allcollisions-nv` 23.6k. Worth picking the light one,
not worth agonising over.

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
| ~~M0~~ | ~~Throughput harness~~ | ✅ ~58 k steps/s — section 1 |
| **M1** | Vectorized env in workers + the four seams | `alpha_stand` replay scores as expected |
| **M2** | PPO on CPU + checkpoint/resume | Toy task solved, then "hold the pose" |
| **M3** | **Rollout inference first** (WASM SIMD GEMM), then the WebGPU learner — M0 says inference is 68% of the budget | Iteration time low enough to watch |
| **M4** | Fine-tune from a shipped checkpoint | A visibly adapted policy |
| **M5** | From-scratch standup, ONNX export, round-trip into Simulate | A policy we trained, running in the Simulate tab |

**M4 before M5 on purpose.** Fine-tuning converges in far fewer steps and is
the most likely thing to actually work in a tab. M0 came in above estimate, so
M5 now looks reachable rather than aspirational — but the ordering stands,
because a 56-minute run is still a bad debugging loop.

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
