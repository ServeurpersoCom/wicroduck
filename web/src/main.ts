// Wires the sim, the policy and the viewer together and drives them from the
// browser's animation frame clock.

import "./style.css";
import { loadSimulation } from "./sim/scene";
import { Policy } from "./sim/policy";
import { MicroduckController, type Phase } from "./sim/controller";
import { CTRL_DT, TRUNK_BODY } from "./sim/microduck";
import { Viewer } from "./render/viewer";
import { assetUrl } from "./asset-url";

const POLICY_URL = assetUrl("policies/alpha_stand.onnx");

/** Never advance more than this much sim time per frame: after a tab switch
 *  the elapsed time can be seconds, and catching up would freeze the page. */
const MAX_CATCHUP_S = 0.25;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const PHASE_LABEL: Record<Phase, string> = {
  standing: "Standing",
  limp: "Falling",
  settling: "Settling",
  recovering: "Getting up",
};

async function main(): Promise<void> {
  const loader = $("loader");
  const loaderText = $("loader-text");
  const loaderBar = $("loader-bar");

  const viewer = new Viewer($<HTMLCanvasElement>("view"));
  // Render the (empty) scene straight away so the loader sits over something
  // rather than a blank canvas that flashes at first paint.
  viewer.render();

  const sim = await loadSimulation((stage, done, total) => {
    loaderText.textContent = total ? `${stage} ${done}/${total}` : stage;
    loaderBar.style.width = total ? `${((done ?? 0) / total) * 100}%` : "0%";
  });

  loaderText.textContent = "Loading policy";
  loaderBar.style.width = "100%";
  const policy = await Policy.load(POLICY_URL);

  const trunkId = sim.mujoco.mj_name2id(
    sim.model, sim.mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY,
  );
  viewer.build(sim.model, { followBody: trunkId });
  viewer.sync(sim.model, sim.data);

  const controller = new MicroduckController(sim, policy);
  loader.hidden = true;

  // ── Controls ──────────────────────────────────────────────────────────
  const loopToggle = $<HTMLInputElement>("chk-loop");
  for (const id of ["btn-knock", "btn-push", "btn-stand", "btn-reset"]) {
    $<HTMLButtonElement>(id).disabled = false;
  }
  $("btn-knock").addEventListener("click", () => controller.knockDown());
  $("btn-push").addEventListener("click", () => controller.push());
  $("btn-stand").addEventListener("click", () => controller.standUp());
  $("btn-reset").addEventListener("click", () => {
    loopToggle.checked = false;
    controller.reset();
  });
  $<HTMLInputElement>("chk-collision").addEventListener("change", (e) => {
    viewer.setCollisionVisible((e.target as HTMLInputElement).checked);
  });

  // Auto-repeat: once a get-up settles, wait a beat and knock the duck over
  // again, so the demo runs unattended.
  let loopTimer: ReturnType<typeof setTimeout> | undefined;
  controller.onRecoveryEnd = () => {
    if (!loopToggle.checked) return;
    loopTimer = setTimeout(() => controller.knockDown(), 1200);
  };
  loopToggle.addEventListener("change", () => {
    if (loopToggle.checked) controller.knockDown();
    else clearTimeout(loopTimer);
  });

  // ── Loop ──────────────────────────────────────────────────────────────
  // Physics is stepped on a fixed 50 Hz control clock decoupled from the
  // display refresh, so the duck behaves the same on a 60 Hz laptop and a
  // 144 Hz monitor.
  const tPhase = $("t-phase"), tGz = $("t-gz"), tHeight = $("t-height"), tRate = $("t-rate");
  let last = performance.now() / 1000;
  let accumulator = 0;
  let stepsThisSecond = 0;
  let rateWindow = 0;
  let displayedRate = 0;

  // One control step at a time: inference is async, and overlapping runs
  // would feed the policy a stale observation.
  const loop = async (): Promise<void> => {
    const now = performance.now() / 1000;
    const dt = Math.min(now - last, MAX_CATCHUP_S);
    last = now;
    accumulator += dt;

    while (accumulator >= CTRL_DT) {
      accumulator -= CTRL_DT;
      await controller.step();
      stepsThisSecond++;
    }

    rateWindow += dt;
    if (rateWindow >= 0.5) {
      displayedRate = stepsThisSecond / rateWindow;
      stepsThisSecond = 0;
      rateWindow = 0;
    }

    const t = controller.telemetry();
    tPhase.textContent = PHASE_LABEL[t.phase];
    // gravityZ is -1 upright and 0 flat on the floor; show it as a percentage
    // so the number means something without explaining projected gravity.
    tGz.textContent = `${Math.round(Math.max(0, -t.gravityZ) * 100)}%`;
    tHeight.textContent = `${(t.height * 100).toFixed(1)} cm`;
    tRate.textContent = `${displayedRate.toFixed(0)} Hz`;

    viewer.sync(sim.model, sim.data);
    viewer.render();
    requestAnimationFrame(() => void loop());
  };
  void loop();
}

main().catch((err: unknown) => {
  console.error(err);
  const box = document.getElementById("error");
  const loader = document.getElementById("loader");
  if (loader) loader.hidden = true;
  if (box) {
    box.hidden = false;
    box.textContent = `Failed to start:\n\n${err instanceof Error ? err.message : String(err)}`;
  }
});
