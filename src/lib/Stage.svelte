<script lang="ts">
  import { onMount } from "svelte";
  import type { Session } from "./session.svelte";

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
    <div class="overlay error">
      <strong>Failed to start</strong>
      <span>{session.error}</span>
    </div>
  {:else if !session.ready}
    <div class="overlay">
      <div class="spinner"></div>
      <span>{session.loadStage}</span>
      <div class="bar"><div style:width="{session.loadProgress}%"></div></div>
    </div>
  {/if}
</div>

<style>
  .stage { position: relative; min-height: 0; background: var(--viewport); }
  canvas { display: block; width: 100%; height: 100%; }

  .overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    padding: 24px; text-align: center;
    background: color-mix(in srgb, var(--viewport) 92%, transparent);
    font-size: 12px; color: var(--muted);
  }
  .spinner {
    width: 24px; height: 24px; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .bar { width: 200px; height: 2px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width 0.15s ease; }

  .error { color: var(--danger); }
  .error strong { font-size: 13px; }
  .error span { max-width: 46ch; line-height: 1.5; }
</style>
