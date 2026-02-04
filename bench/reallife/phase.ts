/**
 * Phase Benchmark
 *
 * Benchmarks phase computation:
 * - Sine vs cosine phase difference
 * - Known phase signals
 * - spectrum() phase output
 * - DC phase for positive/negative signals
 *
 * Run: bun run bench/reallife/phase.ts
 */

import { FFT, phase } from "../../src/xform/fourier.js";
import { spectrum } from "../../src/public/spectrum.js";
import { BenchContext, loadCosine, loadPureSine } from "./helpers.js";

const bench = new BenchContext("Phase Benchmark");

// =============================================================================
// Sine vs Cosine Phase Difference
// =============================================================================

bench.section("Sine vs Cosine Phase Difference");

const sineRef = loadPureSine();
const cosineRef = loadCosine();
const fft = new FFT(sineRef.n);

// Find matching sine and cosine cases at bin 8
const sineBin8 = sineRef.cases.find(
  (c) =>
    c.kind === "pure_sine_bin_centered" &&
    c.params.bin_index === 8 &&
    c.params.amplitude === 1.0
);
const cosineBin8 = cosineRef.cases.find(
  (c) => c.kind === "cosine" && c.params.bin_index === 8
);

if (sineBin8 && cosineBin8) {
  const sineInput = Float64Array.from(sineBin8.signal);
  const cosineInput = Float64Array.from(cosineBin8.signal);
  const bin = sineBin8.params.bin_index as number;

  // Benchmark phase computation for sine
  bench.time(`FFT + phase (sine_bin${bin}, n=${sineBin8.n})`, () => {
    const result = fft.forward(sineInput);
    return phase(result);
  });

  // Benchmark phase computation for cosine
  bench.time(`FFT + phase (cosine_bin${bin}, n=${cosineBin8.n})`, () => {
    const result = fft.forward(cosineInput);
    return phase(result);
  });

  // Verify 90-degree phase difference
  const sineResult = fft.forward(sineInput);
  const cosineResult = fft.forward(cosineInput);
  const sinePhase = phase(sineResult);
  const cosinePhase = phase(cosineResult);

  let phaseDiff = cosinePhase[bin]! - sinePhase[bin]!;
  // Normalize to [-π, π]
  while (phaseDiff > Math.PI) phaseDiff -= 2 * Math.PI;
  while (phaseDiff < -Math.PI) phaseDiff += 2 * Math.PI;

  // Cosine should lead by π/2
  const expectedDiff = Math.PI / 2;
  const diffError = Math.abs(phaseDiff - expectedDiff);
  if (diffError > 1e-6) {
    console.warn(
      `  WARNING: Phase difference ${phaseDiff.toFixed(6)}, expected ${expectedDiff.toFixed(6)}, error ${diffError.toFixed(6)}`
    );
  } else {
    console.log(
      `  VERIFIED: Cosine leads sine by ${(phaseDiff * 180 / Math.PI).toFixed(2)} degrees at bin ${bin}`
    );
  }
}

bench.memory("After sine/cosine phase");

// =============================================================================
// Known Phase Signals
// =============================================================================

bench.section("Known Phase Signals");

const phaseCases = sineRef.cases.filter((c) => c.kind === "pure_sine_phase");

for (const testCase of phaseCases) {
  const input = Float64Array.from(testCase.signal);
  const bin = testCase.params.bin_index as number;
  const expectedPhase = testCase.phase[bin]!;

  const ph = bench.time(`FFT + phase (${testCase.name})`, () => {
    const result = fft.forward(input);
    return phase(result);
  });

  // Verify phase matches NumPy reference
  let diff = Math.abs(ph[bin]! - expectedPhase);
  // Handle phase wrapping
  diff = Math.min(diff, Math.abs(diff - 2 * Math.PI));

  if (diff > 1e-10) {
    console.warn(
      `  WARNING: Phase at bin ${bin} is ${ph[bin]}, expected ${expectedPhase}, diff ${diff}`
    );
  }
}

bench.memory("After known phase");

// =============================================================================
// spectrum() Phase Output
// =============================================================================

bench.section("spectrum() Phase Output");

for (const testCase of phaseCases) {
  const input = Float64Array.from(testCase.signal);
  const expectedBin = testCase.params.bin_index as number;
  const expectedPhase = testCase.phase[expectedBin]!;

  const result = bench.time(`spectrum (${testCase.name})`, () =>
    spectrum(input, {
      sampleRate: testCase.sampleRate,
      fftSize: testCase.n,
      window: "rect",
      sides: "one"
    })
  );

  // Verify peak index
  if (result.peak.index !== expectedBin) {
    console.warn(
      `  WARNING: Peak at bin ${result.peak.index}, expected ${expectedBin}`
    );
  }

  // Verify peak phase
  let diff = Math.abs(result.peak.phase - expectedPhase);
  diff = Math.min(diff, Math.abs(diff - 2 * Math.PI));
  if (diff > 1e-10) {
    console.warn(
      `  WARNING: Peak phase ${result.peak.phase}, expected ${expectedPhase}, diff ${diff}`
    );
  }
}

bench.memory("After spectrum() phase");

// =============================================================================
// Phase Array Length
// =============================================================================

bench.section("Phase Array Length");

const testCase = sineRef.cases[0]!;
const input = Float64Array.from(testCase.signal);

// One-sided
const oneSidedResult = bench.time(
  `spectrum (one-sided, n=${testCase.n})`,
  () =>
    spectrum(input, {
      sampleRate: testCase.sampleRate,
      fftSize: testCase.n,
      window: "rect",
      sides: "one"
    })
);

const expectedOneSided = testCase.n / 2 + 1;
if (oneSidedResult.phase.length !== expectedOneSided) {
  console.warn(
    `  WARNING: One-sided phase length ${oneSidedResult.phase.length}, expected ${expectedOneSided}`
  );
} else {
  console.log(
    `  VERIFIED: One-sided phase length = ${oneSidedResult.phase.length} (N/2 + 1)`
  );
}

// Two-sided
const twoSidedResult = bench.time(
  `spectrum (two-sided, n=${testCase.n})`,
  () =>
    spectrum(input, {
      sampleRate: testCase.sampleRate,
      fftSize: testCase.n,
      window: "rect",
      sides: "two"
    })
);

if (twoSidedResult.phase.length !== testCase.n) {
  console.warn(
    `  WARNING: Two-sided phase length ${twoSidedResult.phase.length}, expected ${testCase.n}`
  );
} else {
  console.log(
    `  VERIFIED: Two-sided phase length = ${twoSidedResult.phase.length} (N)`
  );
}

bench.memory("After phase array length");

// =============================================================================
// DC Phase for Symmetric Signal
// =============================================================================

bench.section("DC Phase (Positive/Negative Signals)");

const n = 64;
const dcFft = new FFT(n);

// Positive DC signal
const positiveDc = new Float64Array(n).fill(1.0);
const posResult = bench.time(`FFT + phase (positive DC, n=${n})`, () => {
  const result = dcFft.forward(positiveDc);
  return phase(result);
});

if (Math.abs(posResult[0]!) > 1e-10) {
  console.warn(`  WARNING: Positive DC phase ${posResult[0]}, expected ~0`);
} else {
  console.log(`  VERIFIED: Positive DC phase = ${posResult[0]!.toFixed(6)}`);
}

// Negative DC signal
const negativeDc = new Float64Array(n).fill(-1.0);
const negResult = bench.time(`FFT + phase (negative DC, n=${n})`, () => {
  const result = dcFft.forward(negativeDc);
  return phase(result);
});

// atan2(0, negative) = π
if (Math.abs(Math.abs(negResult[0]!) - Math.PI) > 1e-10) {
  console.warn(`  WARNING: Negative DC phase ${negResult[0]}, expected ~π`);
} else {
  console.log(`  VERIFIED: Negative DC phase = ${negResult[0]!.toFixed(6)} (≈ π)`);
}

bench.memory("After DC phase");

// =============================================================================
// Phase Computation on Various Signal Sizes
// =============================================================================

bench.section("Phase Computation (Various Sizes)");

const sizes = [64, 128, 256, 512, 1024, 2048, 4096];

for (const size of sizes) {
  // Generate a simple sine wave for each size
  const k = 8;
  const freq = (k * sineRef.sampleRate) / size;
  const signal = new Float64Array(size);
  for (let i = 0; i < size; i += 1) {
    const t = i / sineRef.sampleRate;
    signal[i] = Math.sin(2 * Math.PI * freq * t);
  }

  const sizeFft = new FFT(size);

  bench.time(`FFT + phase (n=${size})`, () => {
    const result = sizeFft.forward(signal);
    return phase(result);
  });
}

bench.memory("After various sizes");

// =============================================================================
// Batch Phase Processing
// =============================================================================

bench.section("Batch Phase Processing (100 frames)");

if (sineBin8) {
  const batchInput = Float64Array.from(sineBin8.signal);
  const batchFft = new FFT(sineRef.n);

  bench.time(
    `100 FFT + phase (n=${sineRef.n})`,
    () => {
      let lastPhase;
      for (let i = 0; i < 100; i += 1) {
        const result = batchFft.forward(batchInput);
        lastPhase = phase(result);
      }
      return lastPhase;
    },
    { iterations: 10 }
  );

  bench.time(
    `100 spectrum with phase (n=${sineRef.n})`,
    () => {
      let lastResult;
      for (let i = 0; i < 100; i += 1) {
        lastResult = spectrum(batchInput, {
          sampleRate: sineRef.sampleRate,
          fftSize: sineRef.n,
          window: "rect",
          sides: "one"
        });
      }
      return lastResult;
    },
    { iterations: 10 }
  );
}

bench.memory("After batch processing");

// =============================================================================
// Report
// =============================================================================

bench.report();
