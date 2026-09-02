// Assembles the MuJoCo scene the app simulates: the robot MJCF from the
// microduck_rl submodule, wrapped in a world that adds a floor, pins the
// timestep the policies were trained at, and declares the STAND keyframe.

import { loadMujoco, type MjData, type MjModel, type Mujoco } from "./mujoco";
import { DEFAULT_POSE, TIMESTEP } from "./microduck";
import { assetUrl } from "../asset-url";

const MODEL_DIR = assetUrl("model/microduck");

interface Manifest {
  xml: string[];
  meshes: string[];
}

export interface Progress {
  (stage: string, done?: number, total?: number): void;
}

export interface Simulation {
  mujoco: Mujoco;
  model: MjModel;
  data: MjData;
  /** Index of the STAND keyframe, for mj_resetDataKeyframe. */
  standKey: number;
}

/**
 * The robot MJCF has no world around it: no floor, no keyframe, and a 2 ms
 * timestep that does not match the 5 ms the policies were trained with. This
 * wraps it into a scene that fixes all three.
 *
 * The `<include>` is resolved by the MuJoCo compiler against the VFS, so the
 * robot XML and its meshes just have to be present under the same names the
 * MJCF uses (`meshdir="assets"`).
 */
function sceneXml(robotXml: string): string {
  const pose = Array.from(DEFAULT_POSE).join(" ");
  return `<mujoco model="microduck-scene">
  <include file="${robotXml}"/>
  <option timestep="${TIMESTEP}"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.28 0.3 0.34 1"/>
  </worldbody>
  <keyframe>
    <!-- STAND2, the pose the policies treat as zero: free joint (7) then the
         14 hinges in document order, which is the actuator order. -->
    <key name="STAND" qpos="0 0 0.12 1 0 0 0 ${pose}" ctrl="${pose}"/>
  </keyframe>
</mujoco>`;
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function loadSimulation(onProgress: Progress = () => {}): Promise<Simulation> {
  onProgress("Loading MuJoCo runtime");
  const [mujoco, manifest] = await Promise.all([
    loadMujoco(),
    fetch(`${MODEL_DIR}/manifest.json`).then((r) => {
      if (!r.ok) throw new Error(`manifest.json -> HTTP ${r.status}; run \`npm run prepare-assets\``);
      return r.json() as Promise<Manifest>;
    }),
  ]);

  const vfs = new mujoco.MjVFS();

  onProgress("Loading robot model", 0, manifest.meshes.length);
  await Promise.all(
    manifest.xml.map(async (f) => vfs.addBuffer(f, await fetchBytes(`${MODEL_DIR}/${f}`))),
  );

  // The collision geoms are meshes too, so every mesh is needed to compile —
  // there is no visual-only subset to skip.
  let loaded = 0;
  await Promise.all(
    manifest.meshes.map(async (f) => {
      const buf = await fetchBytes(`${MODEL_DIR}/assets/${f}`);
      vfs.addBuffer(`assets/${f}`, buf);
      onProgress("Loading robot model", ++loaded, manifest.meshes.length);
    }),
  );

  onProgress("Compiling physics");
  const model = mujoco.MjModel.from_xml_string(sceneXml(manifest.xml[0]), vfs);
  const data = new mujoco.MjData(model);
  const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
  mujoco.mj_resetDataKeyframe(model, data, standKey);
  mujoco.mj_forward(model, data);

  return { mujoco, model, data, standKey };
}
