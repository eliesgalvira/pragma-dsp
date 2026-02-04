import { bench, group, run } from "mitata";
import { FFT } from "../src/xform/fourier.js";
import { loadFixtures } from "../test/fixtures.js";

const fixtures = loadFixtures();
const benchCases = fixtures.fftCases.filter(
  (c) => c.kind === "benchmark_random_normal"
);

const checksums = new Map<number, number>();

for (const fixture of benchCases) {
  const fft = new FFT(fixture.n);
  const input = Float64Array.from(fixture.input);
  const out = fft.createComplexArray();

  group(`fft n=${fixture.n}`, () => {
    bench("forward", () => {
      const result = fft.forward(input, out);
      let checksum = 0;
      for (let i = 0; i < fixture.n; i += 1) {
        checksum += (result.real[i] ?? 0) * 0.001;
        checksum += (result.imag[i] ?? 0) * 0.002;
      }
      checksums.set(fixture.n, checksum);
    });
  });
}

await run();

for (const [n, checksum] of checksums) {
  // Guardrail output to ensure deterministic work per run.
  console.log(`checksum n=${n}: ${checksum.toFixed(6)}`);
}
