// Checkpoint persistence in the Origin Private File System.
//
// OPFS rather than localStorage because a checkpoint is megabytes of weights
// and Adam moments, and rather than IndexedDB because the API is a plain file
// handle. Available on the main thread and in workers, which matters: the
// trainer runs in a worker and should be able to autosave without a round
// trip.
//
// A run that cannot survive a reload is a run you cannot leave overnight, so
// this is M2 work rather than a later nicety.

const DIR = "checkpoints";

async function dir(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

export function opfsAvailable(): boolean {
  return typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function";
}

export async function saveCheckpoint(name: string, data: unknown): Promise<void> {
  const handle = await (await dir()).getFileHandle(`${name}.json`, { create: true });
  const writable = await handle.createWritable();
  try {
    await writable.write(JSON.stringify(data));
  } finally {
    await writable.close();
  }
}

export async function loadCheckpoint<T>(name: string): Promise<T | null> {
  try {
    const handle = await (await dir()).getFileHandle(`${name}.json`);
    return JSON.parse(await (await handle.getFile()).text()) as T;
  } catch {
    return null; // absent is a normal answer, not an error
  }
}

export interface CheckpointInfo {
  name: string;
  bytes: number;
  modified: number;
}

export async function listCheckpoints(): Promise<CheckpointInfo[]> {
  const out: CheckpointInfo[] = [];
  try {
    const d = await dir() as FileSystemDirectoryHandle & {
      values(): AsyncIterableIterator<FileSystemHandle>;
    };
    for await (const entry of d.values()) {
      if (entry.kind !== "file" || !entry.name.endsWith(".json")) continue;
      const file = await (entry as FileSystemFileHandle).getFile();
      out.push({ name: entry.name.replace(/\.json$/, ""), bytes: file.size, modified: file.lastModified });
    }
  } catch {
    return [];
  }
  return out.sort((a, b) => b.modified - a.modified);
}

export async function deleteCheckpoint(name: string): Promise<void> {
  try {
    await (await dir()).removeEntry(`${name}.json`);
  } catch {
    /* already gone */
  }
}
