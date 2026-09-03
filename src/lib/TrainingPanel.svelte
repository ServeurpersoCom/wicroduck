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
      <h3>Train a policy</h3>
      <p>
        PPO over the vectorized environment, in a worker. Pick a task, press
        Train, and watch the standing line — reward can climb while the
        behaviour gets worse. Save a run to replay it in Simulate or continue
        it later.
      </p>
    </div>
    <div class="buttons">
      {#if busy}
        <button onclick={() => s.saveAs()} disabled={!s.iteration}>Save as…</button>
        <button onclick={() => s.stop()}>Stop</button>
      {:else}
        <button class="primary" onclick={() => void s.start(false)}>Train</button>
        {#if s.iteration}
          <button onclick={() => s.saveAs()}>Save as…</button>
        {/if}
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

  {#if s.checkpoints.length}
    <div class="runs">
      <h4>Saved runs</h4>
      <p class="runs-hint">
        Pick one to continue with <strong>Resume</strong>; any of them can also
        be selected as the driving policy in <strong>Simulate</strong>.
      </p>
      <ul>
        {#each s.checkpoints as c (c.name)}
          <li class:picked={(s.resumeFrom ?? "") === c.name}>
            <button
              class="pick"
              disabled={busy}
              onclick={() => (s.resumeFrom = s.resumeFrom === c.name ? null : c.name)}
            >{c.name}</button>
            <span class="meta">
              {(c.bytes / 1024 ** 2).toFixed(1)} MB ·
              {new Date(c.modified).toLocaleTimeString()}
            </span>
            <button class="del" disabled={busy} onclick={() => void s.remove(c.name)}>Delete</button>
          </li>
        {/each}
      </ul>
    </div>
  {/if}

  <p class="foot">
    {#if !s.opfs}
      Checkpoints unavailable — this browser has no OPFS.
    {:else if s.checkpoints.length}
      {#if s.savedAt}
        Saved “{s.savedName}” at iteration {s.savedAt}
      {:else}
        Checkpoint available
      {/if}
      ({(s.checkpoints[0].bytes / 1024 ** 2).toFixed(1)} MB), autosaving every 25.
      {#if s.resumedAt}Resumed from iteration {s.resumedAt}.{/if}
    {:else}
      Autosaves to the browser's private filesystem every 25 iterations, so a
      run survives a reload.
    {/if}
    {#if s.paramCount}
      · {s.simd ? "SIMD kernels" : "JavaScript fallback"}
    {/if}
  </p>
</section>

<style>
  .train-panel { display: flex; flex-direction: column; gap: 10px; }
  .sub { display: flex; align-items: flex-start; gap: 20px; }
  .sub > div:first-child { flex: 1; min-width: 0; }
  .buttons { display: flex; gap: 6px; }
  h3 { margin: 0; font-size: 13px; letter-spacing: -0.01em; }
  p { margin: 6px 0 0; font-size: 12px; line-height: 1.55; color: var(--muted); max-width: 78ch; }

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

  .runs { display: flex; flex-direction: column; gap: 6px; }
  h4 {
    margin: 0; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  }
  .runs-hint { margin: 0; font-size: 11px; }
  .runs ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 3px; }
  .runs li {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 8px; border-radius: 6px;
    border: 1px solid transparent; font-size: 11px;
  }
  .runs li.picked { border-color: var(--accent); background: var(--panel); }
  .pick {
    padding: 0; border: 0; background: none; font: inherit; font-weight: 600;
    color: var(--ink); cursor: pointer;
  }
  .pick:hover { color: var(--accent); }
  .meta { color: var(--muted); font-variant-numeric: tabular-nums; }
  .del {
    margin-left: auto; padding: 2px 8px; font-size: 10px; font-weight: 500;
    color: var(--muted); background: none; border: 1px solid var(--line);
  }
  .del:hover { color: var(--danger); border-color: var(--danger); }
  .error { color: var(--danger); font-size: 12px; }
</style>
