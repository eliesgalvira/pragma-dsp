import { defineConfig } from "tsdown";

export default defineConfig({
  entry: [
    "./src/index.ts",
    "./src/xform/index.ts",
    "./src/xform/fourier.ts",
    "./src/xform/fourier-fluent.ts",
    "./src/core/index.ts",
    "./src/effect/index.ts",
    "./src/math/index.ts",
    "./src/fluent/index.ts"
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  minify: true
});
