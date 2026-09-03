<script lang="ts">
  import { PHASE_LABEL, POLICY_NAME, type Session } from "./session.svelte";

  const { session }: { session: Session } = $props();
</script>

<footer class="status">
  <!-- data-phase drives the dot colour; "loading" leaves it neutral rather
       than showing the initial "standing" green before the sim exists. -->
  <span class="phase" data-phase={session.ready ? session.phase : "loading"}>
    <i></i>{session.started ? (session.ready ? PHASE_LABEL[session.phase] : "Loading") : "Idle"}
  </span>
  {#if session.started}
    <span><b>Upright</b>{session.uprightPct}%</span>
    <span><b>Trunk</b>{session.heightCm.toFixed(1)} cm</span>
    <span><b>Policy</b>{session.policyHz} Hz</span>
  {:else}
    <span class="dim">Simulator loads on first visit</span>
  {/if}
  <span class="spacer"></span>
  <span class="dim">MuJoCo WASM · onnxruntime-web · {POLICY_NAME}</span>
</footer>

<style>
  .status {
    grid-area: status;
    display: flex; align-items: center; gap: 16px;
    height: 26px; padding: 0 12px;
    background: var(--panel); border-top: 1px solid var(--line);
    font-size: 11px; font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .status span { display: flex; align-items: center; gap: 6px; }
  b { font-weight: 500; color: var(--muted); }

  .phase { font-weight: 600; }
  i { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
  .phase[data-phase="standing"] i { background: var(--ok); }
  .phase[data-phase="recovering"] i { background: var(--accent); }
  .phase[data-phase="limp"] i,
  .phase[data-phase="settling"] i { background: var(--warn); }

  .spacer { flex: 1; }
  .dim { color: var(--muted); overflow: hidden; text-overflow: ellipsis; }
  @media (max-width: 860px) { .dim { display: none; } }
</style>
