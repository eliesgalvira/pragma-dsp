import { defineConfig } from "tsdown";

export default defineConfig({
  entry: [
    "./src/index.ts",
    "./src/xform/index.ts",
    "./src/xform/fourier.ts",
    "./src/xform/stft.ts",
    "./src/xform/fourier-fluent.ts",
    "./src/core/index.ts",
    "./src/effect/index.ts",
    "./src/effect/Fourier.ts",
    "./src/math/index.ts",
    "./src/fluent/index.ts",
    "./src/analysis/index.ts"
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  minify: true
});
