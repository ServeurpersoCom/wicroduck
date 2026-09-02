<script lang="ts">
  import type { Session } from "./session.svelte";

  const { session }: { session: Session } = $props();
</script>

<footer class="controls">
  <button class="primary" disabled={!session.ready} onclick={() => session.knockDown()}>
    Knock down &amp; stand up
  </button>
  <button disabled={!session.ready} onclick={() => session.push()}>Push</button>
  <button disabled={!session.ready} onclick={() => session.standUp()}>Stand up</button>
  <button disabled={!session.ready} onclick={() => session.reset()}>Reset</button>

  <label class="toggle">
    <!-- Not `bind:` + a separate onchange: the two handlers race, and the
         side effect would read a stale value if it won. One handler owns it. -->
    <input
      type="checkbox"
      checked={session.autoRepeat}
      onchange={(e) => session.setAutoRepeat(e.currentTarget.checked)}
      disabled={!session.ready}
    />
    Auto-repeat
  </label>
  <label class="toggle">
    <input type="checkbox" bind:checked={session.showCollision} disabled={!session.ready} />
    Collision geoms
  </label>
</footer>

<style>
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }

  button {
    font: inherit; font-size: 13px; font-weight: 600;
    color: var(--ink); background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px;
    padding: 9px 14px; cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: #38455c; background: #1a1f2b; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover:not(:disabled) { background: #ffd668; }

  .toggle { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); }
  .toggle input { accent-color: var(--accent); }
  .toggle:has(input:disabled) { opacity: 0.45; }
</style>
