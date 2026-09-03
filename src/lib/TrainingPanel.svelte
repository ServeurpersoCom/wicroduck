<script lang="ts">
  import { onMount } from "svelte";
  import { TrainingSession } from "./training.svelte";

  const s = new TrainingSession();
  onMount(() => {
    void s.refreshCheckpoints();
    return () => s.dispose();
  });

  const num = (n: number) => Math.round(n).toLocaleString();
  const busy = $derived(s.status === "running" || s.status === "loading");

  /**
   * Reward and standing-fraction sparkline, drawn as one SVG path each.
   * Two y-scales: reward is unbounded and task-dependent, standing is 0-1.
   */
  const W = 720, H = 120, PAD = 4;
  const path = $derived.by(() => {
    const h = s.history;
    if (h.length < 2) return { reward: "", standing: "", min: 0, max: 0 };
    const min = Math.min(...h.map((p) => p.reward));
    const max = Math.max(...h.map((p) => p.reward));
    const x = (i: number) => PAD + (i / (h.length - 1)) * (W - 2 * PAD);
    const line = (pick: (p: (typeof h)[number]) => number, lo: number, hi: number) =>
      h
        .map((p, i) => {
          const t = (pick(p) - lo) / (hi - lo || 1);
          return `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${(H - PAD - t * (H - 2 * PAD)).toFixed(1)}`;
        })
        .join("");
    return {
      reward: line((p) => p.reward, min, max),
      standing: line((p) => p.standing, 0, 1),
      min,
      max,
    };
  });
</script>

<section class="train-panel">
  <header class="sub">
    <div>
      <h3>Training <span class="tag">M2</span></h3>
      <p>
        PPO over the vectorized environment, in a worker. Correctness is settled
        headlessly — <code>check:grad</code> finite-differences the backprop,
        <code>check:ppo</code> solves a toy task, and <code>check:trainer</code>
        learns to hold the pose and round-trips a checkpoint. This is the same
        loop, watchable.
      </p>
    </div>
    <div class="buttons">
      {#if busy}
        <button onclick={() => s.stop()}>Stop</button>
      {:else}
        <button class="primary" onclick={() => void s.start(false)}>Train</button>
        {#if s.checkpoints.length}
          <button onclick={() => void s.start(true)}>Resume</button>
        {/if}
      {/if}
    </div>
  </header>

  <div class="config">
    <label>
      Task
      <select bind:value={s.task} disabled={busy}>
        <option value="hold_pose">Hold the pose</option>
        <option value="standup">Stand up</option>
      </select>
    </label>
    <label>
      Net
      <select bind:value={s.hiddenPreset} disabled={busy}>
        <option value="small">128/64 (fast)</option>
        <option value="reference">512/256/128 (reference)</option>
      </select>
    </label>
    <label>
      Envs
      <select bind:value={s.envs} disabled={busy}>
        {#each [16, 32, 64, 128] as n (n)}<option value={n}>{n}</option>{/each}
      </select>
    </label>
    <label>
      Iterations
      <select bind:value={s.iterations} disabled={busy}>
        {#each [50, 250, 1000, 5000] as n (n)}<option value={n}>{n}</option>{/each}
      </select>
    </label>
  </div>

  {#if s.error}
    <p class="error">{s.error}</p>
  {/if}

  {#if s.history.length > 1}
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" aria-label="Training progress">
      <path d={path.standing} class="standing" />
      <path d={path.reward} class="reward" />
    </svg>
    <div class="legend">
      <span class="key reward">Reward / step (exploring)</span>
      <span class="range">{path.min.toFixed(2)} – {path.max.toFixed(2)}</span>
      <span class="key standing">Standing</span>
      <span class="range">0 – 100%</span>
    </div>
  {/if}

  {#if s.last}
    {@const l = s.last}
    <dl class="stats">
      <div><dt>Iteration</dt><dd>{l.iteration}</dd></div>
      <div><dt>Steps</dt><dd>{num(l.totalSteps)}</dd></div>
      <!-- Rollout numbers come from the EXPLORING policy, so they read lower
           than the policy is: the deterministic mean does noticeably better.
           Labelled rather than silently flattering. -->
      <div><dt>Reward / step <small>exploring</small></dt><dd>{l.rewardPerStep.toFixed(3)}</dd></div>
      <div><dt>Standing <small>exploring</small></dt><dd>{(l.standingFraction * 100).toFixed(1)}%</dd></div>
      <div><dt>Episode return</dt><dd>{l.episodeReturn.toFixed(1)}</dd></div>
      <div><dt>KL</dt><dd>{l.approxKl.toFixed(4)}</dd></div>
      <div><dt>LR</dt><dd>{l.lr.toExponential(1)}</dd></div>
      <div><dt>Iter time</dt><dd>{(l.rolloutMs + l.updateMs).toFixed(0)} ms</dd></div>
      <div><dt>Rollout / update</dt><dd>{Math.round((l.rolloutMs / (l.rolloutMs + l.updateMs)) * 100)}% / {Math.round((l.updateMs / (l.rolloutMs + l.updateMs)) * 100)}%</dd></div>
      <!-- Measured inside the update, not modelled — see docs/training-plan.md
           on why microbenchmarks mis-price the backward pass. -->
      <div><dt>Fwd / bwd / opt</dt><dd>{l.fwdMs.toFixed(0)} / {l.bwdMs.toFixed(0)} / {l.optMs.toFixed(0)} ms</dd></div>
    </dl>
  {/if}

  <p class="foot">
    {#if !s.opfs}
      Checkpoints unavailable — this browser has no OPFS.
    {:else if s.checkpoints.length}
      {#if s.savedAt}
        Checkpoint saved at iteration {s.savedAt}
      {:else}
        Checkpoint available
      {/if}
      ({(s.checkpoints[0].bytes / 1024 ** 2).toFixed(1)} MB), autosaving every 25.
      {#if s.resumedAt}Resumed from iteration {s.resumedAt}.{/if}
    {:else}
      Autosaves to the browser's private filesystem every 25 iterations, so a
      run survives a reload.
    {/if}
  </p>
</section>

<style>
  .train-panel {
    display: flex; flex-direction: column; gap: 10px;
    padding-top: 14px; border-top: 1px solid var(--line);
  }
  .sub { display: flex; align-items: flex-start; gap: 20px; }
  .sub > div:first-child { flex: 1; min-width: 0; }
  .buttons { display: flex; gap: 6px; }
  h3 { margin: 0; font-size: 13px; letter-spacing: -0.01em; }
  p { margin: 6px 0 0; font-size: 12px; line-height: 1.55; color: var(--muted); max-width: 78ch; }
  code { background: var(--panel-hi); padding: 1px 4px; border-radius: 4px; font-size: 0.92em; }

  button {
    font: inherit; font-size: 12px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 7px;
    padding: 8px 14px; cursor: pointer; white-space: nowrap;
  }
  button:hover { border-color: var(--line-hi); }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover { background: var(--accent-hi); }

  .config { display: flex; flex-wrap: wrap; gap: 8px 16px; }
  .config label {
    display: flex; align-items: center; gap: 6px;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted);
  }
  select {
    font: inherit; font-size: 11px; text-transform: none; letter-spacing: 0;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 5px; padding: 2px 4px;
  }

  svg {
    width: 100%; height: 120px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  }
  path { fill: none; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
  path.reward { stroke: var(--accent); }
  path.standing { stroke: var(--ok); opacity: 0.55; }

  .legend { display: flex; gap: 8px 14px; align-items: center; font-size: 10px; color: var(--muted); }
  .key { display: flex; align-items: center; gap: 5px; }
  .key::before { content: ""; width: 10px; height: 2px; border-radius: 1px; }
  .key.reward::before { background: var(--accent); }
  .key.standing::before { background: var(--ok); }
  .range { font-variant-numeric: tabular-nums; }

  .stats { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 0; font-size: 12px; }
  .stats div { display: flex; flex-direction: column; gap: 2px; }
  .stats dt { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .stats dt small { text-transform: none; letter-spacing: 0; opacity: 0.7; }
  .stats dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }

  .foot { font-size: 10px; color: var(--muted); }
  .error { color: var(--danger); font-size: 12px; }
</style>
