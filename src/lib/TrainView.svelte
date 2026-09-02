<script lang="ts">
  // Placeholder workspace. The roadmap is here rather than in a comment so the
  // gap between "runs a policy" and "trains one" is visible in the product.
  const steps = [
    { done: true, text: "Simulate + run a trained policy in the browser" },
    { done: false, text: "Load any checkpoint, including community ones off the Hub" },
    { done: false, text: "Author environments and reward terms" },
    { done: false, text: "Train with PPO on WebGPU, export to ONNX" },
  ];
</script>

<div class="train">
  <div class="card">
    <span class="badge">Not built yet</span>
    <h2>Training lives here</h2>
    <p>
      The reference stack trains with mjlab + MuJoCo Warp and PPO across
      thousands of parallel environments on a CUDA GPU. Getting that into a tab
      means WebGPU compute and far fewer environments, so this workspace is
      still a placeholder.
    </p>

    <ol>
      {#each steps as step (step.text)}
        <li class:done={step.done}>
          <span class="tick" aria-hidden="true">{step.done ? "✓" : ""}</span>
          {step.text}
        </li>
      {/each}
    </ol>
  </div>
</div>

<style>
  .train {
    grid-area: main;
    display: grid; place-items: center;
    padding: 24px; overflow-y: auto;
    background: var(--viewport);
  }

  .card {
    max-width: 46ch;
    display: flex; flex-direction: column; gap: 12px;
    padding: 22px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  }

  .badge {
    align-self: flex-start;
    padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--line-hi);
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--warn);
  }
  h2 { margin: 0; font-size: 16px; letter-spacing: -0.01em; }
  p { margin: 0; font-size: 12px; line-height: 1.6; color: var(--muted); }

  ol {
    margin: 4px 0 0; padding: 0; list-style: none;
    display: flex; flex-direction: column; gap: 7px;
    font-size: 12px; color: var(--muted);
  }
  li { display: flex; align-items: flex-start; gap: 9px; line-height: 1.45; }
  li.done { color: var(--ink); }
  .tick {
    display: grid; place-items: center; flex: none;
    width: 15px; height: 15px; margin-top: 1px;
    border: 1px solid var(--line-hi); border-radius: 4px;
    font-size: 9px; font-weight: 700;
  }
  li.done .tick { background: var(--ok); border-color: var(--ok); color: #06210f; }
</style>
