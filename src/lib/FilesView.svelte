<script lang="ts">
  import { onMount } from "svelte";
  import {
    deleteCheckpoint, importCheckpoint, listCheckpoints, loadCheckpoint,
    opfsAvailable, readCheckpointFile, renameCheckpoint, sanitizeName,
    type CheckpointInfo,
  } from "../train/checkpoint-store.ts";

  interface Row extends CheckpointInfo {
    iteration?: number;
    task?: string;
    hidden?: string;
  }

  let rows = $state<Row[]>([]);
  let error = $state<string | null>(null);
  let busy = $state(false);
  const available = opfsAvailable();

  async function refresh(): Promise<void> {
    const list = await listCheckpoints();
    // Read each file's header for the details worth showing. A handful of
    // multi-megabyte files, only when this tab is open.
    rows = await Promise.all(
      list.map(async (c) => {
        const data = await loadCheckpoint<{
          iteration?: number; task?: string; config?: { hidden?: number[] };
        }>(c.name);
        return {
          ...c,
          iteration: data?.iteration,
          task: data?.task,
          hidden: data?.config?.hidden?.join("/"),
        };
      }),
    );
  }

  onMount(() => {
    void refresh();
  });

  async function guard(fn: () => Promise<void>): Promise<void> {
    busy = true;
    error = null;
    try {
      await fn();
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      busy = false;
    }
  }

  async function download(name: string): Promise<void> {
    const file = await readCheckpointFile(name);
    const url = URL.createObjectURL(file);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    // Revoked on the next tick — revoking immediately can cancel the download.
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }

  function rename(row: Row): void {
    const raw = globalThis.prompt?.("Rename this run:", row.name);
    if (!raw) return;
    const to = sanitizeName(raw);
    if (!to || to === row.name) return;
    void guard(() => renameCheckpoint(row.name, to));
  }

  function remove(row: Row): void {
    if (!globalThis.confirm?.(`Delete “${row.name}”? This cannot be undone.`)) return;
    void guard(() => deleteCheckpoint(row.name));
  }

  async function upload(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ""; // so the same file can be picked twice
    if (!file) return;
    const name = sanitizeName(file.name.replace(/\.json$/i, ""));
    await guard(async () => importCheckpoint(name || "imported", await file.text()));
  }

  const mb = (b: number) => `${(b / 1024 ** 2).toFixed(1)} MB`;
  const when = (t: number) => new Date(t).toLocaleString();
  const total = $derived(rows.reduce((n, r) => n + r.bytes, 0));
</script>

<div class="files">
  <header>
    <div>
      <h2>Saved runs</h2>
      <p>
        Checkpoints live in this browser's private filesystem — not in a folder
        you can browse, which is why they get a manager. Download one to keep
        it or move it to another machine; upload it back to carry on.
      </p>
    </div>
    <label class="upload">
      <input type="file" accept="application/json,.json" onchange={upload} disabled={busy} />
      <span>Upload…</span>
    </label>
  </header>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if !available}
    <p class="empty">This browser has no private filesystem, so runs cannot be saved.</p>
  {:else if rows.length === 0}
    <p class="empty">
      Nothing saved yet. Train a policy, then use <strong>Save as…</strong> in
      the Train workspace.
    </p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Name</th><th>Task</th><th class="n">Iteration</th>
          <th>Net</th><th class="n">Size</th><th>Modified</th><th></th>
        </tr>
      </thead>
      <tbody>
        {#each rows as r (r.name)}
          <tr>
            <td class="name">{r.name}</td>
            <td>{r.task ?? "—"}</td>
            <td class="n">{r.iteration ?? "—"}</td>
            <td class="mono">{r.hidden ?? "—"}</td>
            <td class="n">{mb(r.bytes)}</td>
            <td class="dim">{when(r.modified)}</td>
            <td class="actions">
              <button disabled={busy} onclick={() => void download(r.name)}>Download</button>
              <button disabled={busy} onclick={() => rename(r)}>Rename</button>
              <button class="danger" disabled={busy} onclick={() => remove(r)}>Delete</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="foot">{rows.length} run{rows.length === 1 ? "" : "s"} · {mb(total)} total</p>
  {/if}
</div>

<style>
  .files {
    grid-area: main;
    display: flex; flex-direction: column; gap: 14px;
    padding: 20px 24px; overflow-y: auto; background: var(--viewport);
  }
  header { display: flex; align-items: flex-start; gap: 20px; }
  header > div { flex: 1; min-width: 0; }
  h2 { margin: 0; font-size: 15px; letter-spacing: -0.01em; }
  p { margin: 6px 0 0; font-size: 12px; line-height: 1.6; color: var(--muted); max-width: 74ch; }

  .upload input { display: none; }
  .upload span {
    display: inline-block; font-size: 12px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 7px;
    padding: 8px 14px; cursor: pointer; white-space: nowrap;
  }
  .upload span:hover { border-color: var(--line-hi); }

  table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 16px 6px 0; text-align: left; white-space: nowrap; }
  th {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted); border-bottom: 1px solid var(--line);
  }
  td { border-bottom: 1px solid color-mix(in srgb, var(--line) 50%, transparent); }
  .n { text-align: right; }
  .name { font-weight: 600; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
  .dim { color: var(--muted); }

  .actions { display: flex; gap: 6px; padding-right: 0; }
  .actions button {
    font: inherit; font-size: 11px; font-weight: 500;
    color: var(--muted); background: none;
    border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 9px; cursor: pointer;
  }
  .actions button:hover:not(:disabled) { color: var(--ink); border-color: var(--line-hi); }
  .actions button.danger:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); }
  .actions button:disabled { opacity: 0.4; cursor: not-allowed; }

  .empty, .foot { font-size: 12px; color: var(--muted); }
  .error { color: var(--danger); font-size: 12px; }
</style>
