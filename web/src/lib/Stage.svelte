<script lang="ts">
  import { onMount } from "svelte";
  import { PHASE_LABEL, type Session } from "./session.svelte";

  const { session }: { session: Session } = $props();

  let canvas: HTMLCanvasElement;

  onMount(() => {
    void session.start(canvas);
    return () => session.destroy();
  });
</script>

<div class="stage">
  <canvas bind:this={canvas}></canvas>

  {#if session.error}
    <div class="error">Failed to start:<br /><br />{session.error}</div>
  {:else if !session.ready}
    <div class="loader">
      <div class="spinner"></div>
      <div>{session.loadStage}</div>
      <div class="bar"><div style:width="{session.loadProgress}%"></div></div>
    </div>
  {/if}

  <dl class="telemetry">
    <div><dt>Phase</dt><dd>{PHASE_LABEL[session.phase]}</dd></div>
    <div><dt>Uprightness</dt><dd>{session.uprightPct}%</dd></div>
    <div><dt>Trunk height</dt><dd>{session.heightCm.toFixed(1)} cm</dd></div>
    <div><dt>Policy</dt><dd>{session.policyHz} Hz</dd></div>
  </dl>
</div>

<style>
  .stage {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    background: #0f1116;
    min-height: 320px;
  }
  canvas { display: block; width: 100%; height: 100%; }

  .telemetry {
    position: absolute; left: 12px; bottom: 12px; margin: 0;
    display: flex; flex-wrap: wrap; gap: 6px 18px;
    background: rgba(11, 13, 18, 0.72);
    border: 1px solid var(--line); border-radius: 8px;
    padding: 8px 12px; backdrop-filter: blur(6px);
    font-variant-numeric: tabular-nums; font-size: 12px;
  }
  .telemetry div { display: flex; flex-direction: column; gap: 2px; }
  .telemetry dt {
    color: var(--muted); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .telemetry dd { margin: 0; font-weight: 600; }

  .loader, .error {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; background: rgba(11, 13, 18, 0.92); font-size: 13px; color: var(--muted);
  }
  .spinner {
    width: 26px; height: 26px; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .bar { width: 220px; height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width 0.15s ease; }
  .error { color: #ff8f8f; padding: 24px; text-align: center; }
</style>
