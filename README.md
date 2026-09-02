# Wicroduck

An all-in-one, **100% in-browser** toolchain for [Microduck](https://github.com/pollen-robotics/microduck) —
the ~800 g, ~25 cm bipedal robot duck. No CUDA box, no Python install, no
backend: open a tab and the whole loop runs there.

The reference training stack lives in [`microduck_rl/`](microduck_rl) (a
submodule of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)):
mjlab + MuJoCo Warp + PPO on a GPU, exported to ONNX. This repo is the attempt
to put that pipeline — simulate, run, and eventually *train* — behind a URL.

## Where it is now

**Step 1 is done: the stand-up demo.** [`web/`](web) is a Svelte + Vite app
that runs MuJoCo compiled to WebAssembly, steps the real Microduck MJCF at
200 Hz, and drives it with the shipped `alpha_stand` ONNX policy at 50 Hz. Knock the duck over and watch it
get itself back on its feet, rendered from the compiled model's own geometry.

```bash
git submodule update --init --recursive
cd web && npm install && npm run dev
```

See [`web/README.md`](web/README.md) for how the policy loop is wired and how
to check it without a browser.

## Where it is going

1. ~~In-browser simulation + inference of a trained policy~~ ✅
2. Load any policy — the other shipped checkpoints, and community ones off the
   Hugging Face Hub
3. Reward / environment authoring in the browser
4. In-browser training (WebGPU), export to ONNX, deploy to a real duck

## Prior art

The stand-up demo is a from-scratch reimplementation, but the approach is not
novel — Pollen ship an official in-browser sandbox built the same way
(MuJoCo WASM + onnxruntime-web), and it is worth playing with:

- [pollen-robotics/microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) — the official sandbox (nine policies, multiplayer ghosts)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) — the shipped ONNX checkpoints
- [awesome-microduck](https://github.com/joeynyc/awesome-microduck) — community simulators, policies and tools
