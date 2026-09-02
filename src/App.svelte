<script lang="ts">
  import Rail from "./lib/Rail.svelte";
  import Stage from "./lib/Stage.svelte";
  import Inspector from "./lib/Inspector.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import TrainView from "./lib/TrainView.svelte";
  import { Session } from "./lib/session.svelte";
  import type { View } from "./lib/views";

  const session = new Session();
  let view = $state<View>("sim");

  // The stage stays mounted across workspace switches — booting MuJoCo and the
  // policy takes seconds, and tearing the WebGL context down to rebuild it
  // would throw that away. Hidden, the physics keeps stepping but the frame is
  // not drawn.
  $effect(() => session.setRendering(view === "sim"));
</script>

<!-- The inspector column only exists for workspaces that have one; without
     this the Train view would sit off-centre next to an empty gutter. -->
<div class="shell" class:no-aside={view !== "sim"}>
  <Rail bind:view />

  <header class="topbar">
    <h1>{view === "sim" ? "Simulate" : "Train"}</h1>
    <span class="crumb">
      {view === "sim" ? "Microduck · stand-up policy" : "Microduck · PPO"}
    </span>
  </header>

  <!-- Kept in the DOM, hidden when another workspace is up front. -->
  <div class="main" class:hidden={view !== "sim"}>
    <Stage {session} />
  </div>
  {#if view === "sim"}
    <Inspector {session} />
  {:else}
    <TrainView />
  {/if}

  <StatusBar {session} />
</div>

<style>
  .shell {
    height: 100%;
    display: grid;
    grid-template-columns: 208px minmax(0, 1fr) 268px;
    grid-template-rows: 44px minmax(0, 1fr) auto;
    grid-template-areas:
      "rail topbar topbar"
      "rail main   aside"
      "rail status status";
  }

  .shell.no-aside {
    grid-template-columns: 208px minmax(0, 1fr);
    grid-template-areas:
      "rail topbar"
      "rail main"
      "rail status";
  }

  .topbar {
    grid-area: topbar;
    display: flex; align-items: center; gap: 10px;
    padding: 0 14px;
    background: var(--panel); border-bottom: 1px solid var(--line);
  }
  h1 { margin: 0; font-size: 13px; font-weight: 700; letter-spacing: -0.01em; }
  .crumb { font-size: 11px; color: var(--muted); }

  .main { grid-area: main; display: grid; min-height: 0; min-width: 0; }
  .main.hidden { display: none; }

  @media (max-width: 1040px) {
    .shell {
      grid-template-columns: 208px minmax(0, 1fr);
      grid-template-rows: 44px minmax(0, 1fr) auto auto;
      grid-template-areas:
        "rail topbar"
        "rail main"
        "rail aside"
        "rail status";
    }
  }
  @media (max-width: 860px) {
    .shell { grid-template-columns: 56px minmax(0, 1fr); }
  }
</style>
