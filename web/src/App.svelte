<script lang="ts">
  import Stage from "./lib/Stage.svelte";
  import Controls from "./lib/Controls.svelte";
  import { Session } from "./lib/session.svelte";

  const session = new Session();
</script>

<div class="app">
  <header>
    <h1>Wicroduck <span class="tag">stand-up demo</span></h1>
    <p>
      MuJoCo (WASM) + the shipped <code>alpha_stand</code> ONNX policy,
      running entirely in this tab.
    </p>
  </header>

  <Stage {session} />
  <Controls {session} />
</div>

<style>
  .app {
    height: 100%;
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: 12px;
    padding: 16px;
    max-width: 1100px;
    margin: 0 auto;
  }

  h1 { margin: 0; font-size: 20px; letter-spacing: -0.01em; }
  .tag {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--accent); border: 1px solid var(--line); border-radius: 999px;
    padding: 2px 8px; margin-left: 6px; vertical-align: 3px;
  }
  p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
  code { background: #1c2230; padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }
</style>
