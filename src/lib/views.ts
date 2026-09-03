/** The workspaces the left rail switches between. */
export type View = "guide" | "sim" | "train" | "files" | "debug";

export const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "guide", label: "Guide", hint: "Start here" },
  { id: "sim", label: "Simulate", hint: "Run a trained policy" },
  { id: "train", label: "Train", hint: "PPO in your browser" },
  { id: "files", label: "Files", hint: "Saved runs" },
  { id: "debug", label: "Debug", hint: "Instruments" },
];
