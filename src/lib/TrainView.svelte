<script lang="ts">
  // Training, and nothing else. The throughput harness and the bare
  // environment rollout live in Debug — they are instruments, not the task.
  import TrainingPanel from "./TrainingPanel.svelte";
</script>

<div class="train">
  <TrainingPanel />
</div>

<style>
  .train {
    grid-area: main;
    display: flex; flex-direction: column;
    padding: 18px 22px; overflow-y: auto; background: var(--viewport);
  }
</style>
