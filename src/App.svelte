<script lang="ts">
  import Rail from "./lib/Rail.svelte";
  import Stage from "./lib/Stage.svelte";
  import Inspector from "./lib/Inspector.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import TrainView from "./lib/TrainView.svelte";
  import GuideView from "./lib/GuideView.svelte";
  import FilesView from "./lib/FilesView.svelte";
  import DebugView from "./lib/DebugView.svelte";
  import { Session } from "./lib/session.svelte";
  import type { View } from "./lib/views.ts";

  const TITLES: Record<View, { title: string; crumb: string }> = {
    guide: { title: "Guide", crumb: "What this is and how to use it" },
    sim: { title: "Simulate", crumb: "Microduck · stand-up policy" },
    train: { title: "Train", crumb: "Microduck · PPO" },
    files: { title: "Files", crumb: "Checkpoints saved in this browser" },
    debug: { title: "Debug", crumb: "Throughput and environment instruments" },
  };

  const session = new Session();
  let view = $state<View>("guide");

  // The stage mounts on the first visit to Simulate and stays mounted after —
  // booting MuJoCo and the policy takes seconds and costs a 21 MB download, so
  // it should not happen behind someone reading the Guide, and it should not
  // happen twice. It is fully paused while hidden: a duck stepping in the
  // background would skew the throughput harness next door.
  let simMounted = $state(false);
  $effect(() => {
    if (view === "sim") {
      simMounted = true;
      // A run saved in the Train workspace after this list was first built
      // should still show up in the policy picker.
      if (session.ready) void session.refreshPolicies();
    }
    session.setActive(view === "sim");
  });
</script>

<!-- The inspector column only exists for workspaces that have one; without
     this the Train view would sit off-centre next to an empty gutter. -->
<div class="shell" class:no-aside={view !== "sim"}>
  <Rail bind:view />

  <header class="topbar">
    <h1>{TITLES[view].title}</h1>
    <span class="crumb">{TITLES[view].crumb}</span>
  </header>

  <!-- Kept in the DOM once mounted, hidden when another workspace is up front. -->
  {#if simMounted}
    <div class="main" class:hidden={view !== "sim"}>
      <Stage {session} />
    </div>
  {/if}
  {#if view === "sim"}
    <Inspector {session} />
  {:else if view === "train"}
    <TrainView />
  {:else if view === "files"}
    <FilesView />
  {:else if view === "debug"}
    <DebugView />
  {:else}
    <GuideView go={(v) => (view = v)} />
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
