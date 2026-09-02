/** The workspaces the left rail switches between. */
export type View = "sim" | "train";

export const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "sim", label: "Simulate", hint: "Run a trained policy" },
  { id: "train", label: "Train", hint: "Throughput harness" },
];
