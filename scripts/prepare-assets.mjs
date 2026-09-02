// Populates public/ with everything the browser app needs at runtime:
//
//   vendor/   MuJoCo + onnxruntime WASM runtimes, copied out of node_modules so
//             they are fetched as plain static files (no bundler in the loop)
//   model/    the Microduck MJCF + the STL meshes it references, from the
//             microduck_rl submodule
//   policies/ the shipped ONNX policies, from the Hugging Face Hub
//
// Everything it writes is gitignored and reproducible: run `npm run
// prepare-assets` (dev/build do it automatically).
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = path.join(ROOT, "public");
const MJCF_SRC = path.join(ROOT, "microduck_rl/src/mjlab_microduck/robot/microduck");

// Policies live in the standalone Apache-2.0 mirror of the checkpoints that
// ship with the robot; the sandbox Space serves the same files.
const POLICY_BASE =
  "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main";
const POLICIES = [
  "alpha_stand.onnx", // get-up / stand-up, the one the demo screen runs
];

const log = (...a) => console.log("[assets]", ...a);

function copyIfChanged(src, dst) {
  const s = fs.statSync(src);
  const d = fs.existsSync(dst) ? fs.statSync(dst) : null;
  if (d && d.size === s.size && d.mtimeMs >= s.mtimeMs) return false;
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  return true;
}

// ── MuJoCo + onnxruntime WASM ────────────────────────────────────────────
function vendorRuntimes() {
  // "exports" hides package.json, so resolve a mapped entry point instead.
  const mjDir = path.dirname(fileURLToPath(import.meta.resolve("@mujoco/mujoco")));
  let n = 0;
  for (const f of ["mujoco.js", "mujoco.wasm"]) {
    n += copyIfChanged(path.join(mjDir, f), path.join(PUBLIC, "vendor/mujoco", f));
  }
  // onnxruntime-web is imported through the bundler, but it fetches its own
  // .wasm/.mjs sidecars at runtime from env.wasm.wasmPaths — those must be
  // static files. Only the non-threaded SIMD build is used: static hosting
  // sends no COOP/COEP, so SharedArrayBuffer (threads) is unavailable.
  const ortDist = path.dirname(
    fileURLToPath(import.meta.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm")),
  );
  for (const f of fs.readdirSync(ortDist)) {
    if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f)) {
      n += copyIfChanged(path.join(ortDist, f), path.join(PUBLIC, "vendor/ort", f));
    }
  }
  log(`runtimes: ${n} file(s) updated`);
}

// ── Microduck MJCF + meshes ──────────────────────────────────────────────
// robot_allcollisions.xml is the variant the standing/sitting policies need:
// it carries shell and body collision geoms (a fallen duck rests its trunk on
// the floor) that the walk-only model leaves out.
const MJCF_FILES = ["robot_allcollisions.xml"];

function vendorModel() {
  if (!fs.existsSync(MJCF_SRC)) {
    throw new Error(
      `microduck_rl submodule not checked out (${MJCF_SRC} missing).\n` +
        "Run: git submodule update --init --recursive",
    );
  }
  const dst = path.join(PUBLIC, "model/microduck");
  let n = 0;
  const meshes = new Set();
  for (const f of MJCF_FILES) {
    const src = path.join(MJCF_SRC, f);
    const xml = fs.readFileSync(src, "utf8");
    for (const m of xml.matchAll(/<mesh[^>]*\bfile="([^"]+)"/g)) meshes.add(m[1]);
    n += copyIfChanged(src, path.join(dst, f));
  }
  for (const mesh of meshes) {
    n += copyIfChanged(path.join(MJCF_SRC, "assets", mesh), path.join(dst, "assets", mesh));
  }
  // The app fetches this instead of walking the XML for <mesh file="..."> a
  // second time in the browser.
  const manifest = { xml: MJCF_FILES, meshes: [...meshes].sort() };
  fs.writeFileSync(path.join(dst, "manifest.json"), JSON.stringify(manifest, null, 2));
  log(`model: ${MJCF_FILES.length} MJCF + ${meshes.size} meshes, ${n} file(s) updated`);
}

// ── ONNX policies ────────────────────────────────────────────────────────
async function fetchPolicies() {
  const dst = path.join(PUBLIC, "policies");
  fs.mkdirSync(dst, { recursive: true });
  for (const name of POLICIES) {
    const out = path.join(dst, name);
    if (fs.existsSync(out) && fs.statSync(out).size > 0) {
      log(`policy: ${name} (cached)`);
      continue;
    }
    const url = `${POLICY_BASE}/${name}`;
    log(`policy: fetching ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    fs.writeFileSync(out, Buffer.from(await res.arrayBuffer()));
  }
}

vendorRuntimes();
vendorModel();
await fetchPolicies();
log("done");
