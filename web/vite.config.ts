import { defineConfig } from "vite";

export default defineConfig({
  // Relative base so a production build also works when served from a
  // subpath (a Hugging Face Space, GitHub Pages, a static folder).
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  resolve: {
    // Without this, onnxruntime-web resolves to its "bundle" build, which
    // base64-inlines a 27 MB .wasm into the JS. The app serves that wasm from
    // public/vendor/ort/ instead (see scripts/prepare-assets.mjs).
    conditions: ["onnxruntime-web-use-extern-wasm"],
  },
  // The MuJoCo runtime is loaded as a static file from public/vendor/, so
  // keep the dep optimizer from trying to pre-bundle it.
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
});
