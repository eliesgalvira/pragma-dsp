/**
 * Scaling Benchmark
 *
 * Benchmarks spectrum() with different scaling modes:
 * - One-sided spectrum (default)
 * - Two-sided spectrum
 * - Frequency axis computation
 * - Peak detection accuracy
 *
 * Run: bun run bench/reallife/scaling.ts
 */

import { spectrum } from "../../src/public/spectrum.js";
import { binFrequencies } from "../../src/xform/fourier.js";
import { BenchContext, loadPureSine, loadSpecial } from "./helpers.js";

const bench = new BenchContext("Scaling Benchmark");

// =============================================================================
// One-Sided Spectrum
// =============================================================================

bench.section("One-Sided Spectrum (sides: 'one')");

const sineRef = loadPureSine();
const binCenteredCases = sineRef.cases.filter(
  (c) => c.kind === "pure_sine_bin_centered"
);

for (const testCase of binCenteredCases.slice(0, 5)) {
  const input = Float64Array.from(testCase.signal);
  const options = {
    sampleRate: testCase.sampleRate,
    fftSize: testCase.n,
    window: "rect" as const,
    sides: "one" as const
  };

  const result = bench.time(
    `spectrum (${testCase.name}, n=${testCase.n})`,
    () => spectrum(input, options)
  );

  // Verify amplitude scaling
  const expectedAmp = testCase.params.amplitude as number;
  const expectedBin = testCase.params.bin_index as number;

  if (result.peak.index !== expectedBin) {
    console.warn(
      `  WARNING: Peak at bin ${result.peak.index}, expected ${expectedBin}`
    );
  }

  const ampError = Math.abs(result.peak.amplitude - expectedAmp);
  if (ampError > 0.01) {
    console.warn(
      `  WARNING: Amplitude ${result.peak.amplitude}, expected ${expectedAmp}, error ${ampError}`
    );
  }
}

bench.memory("After one-sided spectrum");

// =============================================================================
// DC Bin Not Doubled
// =============================================================================

bench.section("DC Bin Scaling");

const specialRef = loadSpecial();
const dcCase = specialRef.cases.find((c) => c.kind === "dc");

if (dcCase) {
  const input = Float64Array.from(dcCase.signal);
  const options = {
    sampleRate: dcCase.sampleRate,
    fftSize: dcCase.n,
    window: "rect" as const,
    sides: "one" as const
  };

  const result = bench.time(`spectrum (DC, n=${dcCase.n})`, () =>
    spectrum(input, options)
  );

  // DC should NOT be doubled
  const expectedDcAmp = dcCase.params.level as number;
  const dcError = Math.abs(result.amplitude[0]! - expectedDcAmp);
  if (dcError > 1e-6) {
    console.warn(
      `  WARNING: DC amplitude ${result.amplitude[0]}, expected ${expectedDcAmp}`
    );
  }
}

// =============================================================================
// Nyquist Bin Not Doubled
// =============================================================================

bench.section("Nyquist Bin Scaling");

const nyquistCase = specialRef.cases.find((c) => c.kind === "nyquist");

if (nyquistCase) {
  const input = Float64Array.from(nyquistCase.signal);
  const options = {
    sampleRate: nyquistCase.sampleRate,
    fftSize: nyquistCase.n,
    window: "rect" as const,
    sides: "one" as const
  };

  const result = bench.time(`spectrum (Nyquist, n=${nyquistCase.n})`, () =>
    spectrum(input, options)
  );

  // Nyquist bin should NOT be doubled
  const nyquistIndex = nyquistCase.n / 2;
  const expectedNyquistAmp = nyquistCase.params.amplitude as number;
  const nyquistError = Math.abs(
    result.amplitude[nyquistIndex]! - expectedNyquistAmp
  );
  if (nyquistError > 1e-6) {
    console.warn(
      `  WARNING: Nyquist amplitude ${result.amplitude[nyquistIndex]}, expected ${expectedNyquistAmp}`
    );
  }
}

bench.memory("After DC/Nyquist scaling");

// =============================================================================
// Two-Sided Spectrum
// =============================================================================

bench.section("Two-Sided Spectrum (sides: 'two')");

for (const testCase of binCenteredCases.slice(0, 5)) {
  const input = Float64Array.from(testCase.signal);
  const options = {
    sampleRate: testCase.sampleRate,
    fftSize: testCase.n,
    window: "rect" as const,
    sides: "two" as const
  };

  const result = bench.time(
    `spectrum (${testCase.name}, n=${testCase.n})`,
    () => spectrum(input, options)
  );

  // Verify two-sided has full N bins
  if (result.amplitude.length !== testCase.n) {
    console.warn(
      `  WARNING: Two-sided has ${result.amplitude.length} bins, expected ${testCase.n}`
    );
  }

  // Verify energy is split between positive and negative frequencies
  const expectedAmp = testCase.params.amplitude as number;
  const expectedBin = testCase.params.bin_index as number;
  const negativeBin = testCase.n - expectedBin;

  // Each side should have amplitude/2
  const posAmpError = Math.abs(result.amplitude[expectedBin]! - expectedAmp / 2);
  const negAmpError = Math.abs(result.amplitude[negativeBin]! - expectedAmp / 2);

  if (posAmpError > 0.01 || negAmpError > 0.01) {
    console.warn(
      `  WARNING: Two-sided amplitude splitting error (pos: ${posAmpError}, neg: ${negAmpError})`
    );
  }
}

bench.memory("After two-sided spectrum");

// =============================================================================
// Frequency Axis Computation
// =============================================================================

bench.section("Frequency Axis (binFrequencies)");

const sizes = [256, 512, 1024, 2048, 4096];
const sampleRate = 48000;

for (const size of sizes) {
  bench.time(`binFrequencies (n=${size}, sides='one')`, () =>
    binFrequencies(size, sampleRate, "one")
  );
}

for (const size of sizes) {
  bench.time(`binFrequencies (n=${size}, sides='two')`, () =>
    binFrequencies(size, sampleRate, "two")
  );
}

// Verify frequency axis correctness
const testSize = 1024;
const freqsOne = binFrequencies(testSize, sampleRate, "one");
const freqsTwo = binFrequencies(testSize, sampleRate, "two");

// One-sided: N/2 + 1 bins
if (freqsOne.length !== testSize / 2 + 1) {
  console.warn(
    `  WARNING: One-sided has ${freqsOne.length} bins, expected ${testSize / 2 + 1}`
  );
}

// Two-sided: N bins
if (freqsTwo.length !== testSize) {
  console.warn(
    `  WARNING: Two-sided has ${freqsTwo.length} bins, expected ${testSize}`
  );
}

// First bin is DC (0 Hz)
if (freqsOne[0] !== 0) {
  console.warn(`  WARNING: First bin is ${freqsOne[0]}, expected 0`);
}

// Last one-sided bin is Nyquist
const nyquist = sampleRate / 2;
if (Math.abs(freqsOne[freqsOne.length - 1]! - nyquist) > 1e-10) {
  console.warn(
    `  WARNING: Last one-sided bin is ${freqsOne[freqsOne.length - 1]}, expected ${nyquist}`
  );
}

bench.memory("After frequency axis");

// =============================================================================
// Peak Detection
// =============================================================================

bench.section("Peak Detection");

// Test that peak detection ignores DC when there are non-DC components
const dcPlusSine = specialRef.cases.find((c) => c.kind === "dc_plus_sine");

if (dcPlusSine) {
  const input = Float64Array.from(dcPlusSine.signal);
  const options = {
    sampleRate: dcPlusSine.sampleRate,
    fftSize: dcPlusSine.n,
    window: "rect" as const,
    sides: "one" as const
  };

  const result = bench.time(`spectrum (DC+sine, n=${dcPlusSine.n})`, () =>
    spectrum(input, options)
  );

  const expectedBin = dcPlusSine.params.sine_bin as number;
  if (result.peak.index !== expectedBin) {
    console.warn(
      `  WARNING: Peak at bin ${result.peak.index}, expected ${expectedBin} (sine bin, not DC)`
    );
  }
}

// Pure DC should return DC as peak
if (dcCase) {
  const input = Float64Array.from(dcCase.signal);
  const options = {
    sampleRate: dcCase.sampleRate,
    fftSize: dcCase.n,
    window: "rect" as const,
    sides: "one" as const
  };

  const result = bench.time(`spectrum peak detection (pure DC)`, () =>
    spectrum(input, options)
  );

  if (result.peak.index !== 0) {
    console.warn(
      `  WARNING: Pure DC peak at bin ${result.peak.index}, expected 0`
    );
  }
}

bench.memory("After peak detection");

// =============================================================================
// Window Types with spectrum()
// =============================================================================

bench.section("Window Types");

const windowTypes = ["rect", "hann", "hamming", "blackman"] as const;
const windowTestCase = binCenteredCases[0]!;
const windowInput = Float64Array.from(windowTestCase.signal);

for (const windowType of windowTypes) {
  const options = {
    sampleRate: windowTestCase.sampleRate,
    fftSize: windowTestCase.n,
    window: windowType,
    sides: "one" as const
  };

  bench.time(`spectrum (${windowType} window, n=${windowTestCase.n})`, () =>
    spectrum(windowInput, options)
  );
}

bench.memory("After window types");

// =============================================================================
// Batch Processing
// =============================================================================

bench.section("Batch Processing (100 spectrum calls)");

const batchInput = Float64Array.from(binCenteredCases[0]!.signal);
const batchOptions = {
  sampleRate: sineRef.sampleRate,
  fftSize: sineRef.n,
  window: "rect" as const,
  sides: "one" as const
};

bench.time(
  `100 spectrum (one-sided, n=${sineRef.n})`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      lastResult = spectrum(batchInput, batchOptions);
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.time(
  `100 spectrum (two-sided, n=${sineRef.n})`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      lastResult = spectrum(batchInput, { ...batchOptions, sides: "two" });
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.memory("After batch processing");

// =============================================================================
// Report
// =============================================================================

bench.report();
