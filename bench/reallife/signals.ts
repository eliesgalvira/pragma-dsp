/**
 * Signals Benchmark
 *
 * Benchmarks FFT forward/inverse operations on various signal types:
 * - Pure sine waves (bin-centered and non-centered)
 * - Multi-tone signals
 * - Chirp signals
 * - Special signals (impulse, DC, Nyquist)
 *
 * Run: bun run bench/reallife/signals.ts
 */

import { FFT, magnitude, phase } from "../../src/xform/fourier.js";
import {
  BenchContext,
  loadChirp,
  loadMultiTone,
  loadPureSine,
  loadSpecial,
  maxAbsError
} from "./helpers.js";

const bench = new BenchContext("Signals Benchmark");

// =============================================================================
// Pure Sine Waves
// =============================================================================

bench.section("Pure Sine Waves - FFT Forward");

const sineRef = loadPureSine();
const sineFft = new FFT(sineRef.n);

// Benchmark a subset of sine cases (to avoid excessive output)
const sineBinCentered = sineRef.cases.filter(
  (c) => c.kind === "pure_sine_bin_centered"
);
const sineNonCentered = sineRef.cases.filter(
  (c) => c.kind === "pure_sine_non_centered"
);
const sinePhase = sineRef.cases.filter((c) => c.kind === "pure_sine_phase");

for (const testCase of sineBinCentered.slice(0, 5)) {
  const input = Float64Array.from(testCase.signal);
  bench.time(`FFT forward (${testCase.name}, n=${testCase.n})`, () => {
    return sineFft.forward(input);
  });
}

for (const testCase of sineNonCentered) {
  const input = Float64Array.from(testCase.signal);
  bench.time(`FFT forward (${testCase.name}, n=${testCase.n})`, () => {
    return sineFft.forward(input);
  });
}

bench.memory("After sine FFT forward");

// =============================================================================
// Pure Sine Waves - Magnitude and Phase
// =============================================================================

bench.section("Pure Sine Waves - Magnitude/Phase");

for (const testCase of sineBinCentered.slice(0, 3)) {
  const input = Float64Array.from(testCase.signal);
  const fftResult = sineFft.forward(input);

  bench.time(`magnitude (${testCase.name})`, () => {
    return magnitude(fftResult);
  });

  bench.time(`phase (${testCase.name})`, () => {
    return phase(fftResult);
  });
}

bench.memory("After magnitude/phase");

// =============================================================================
// Pure Sine Waves - Round-trip
// =============================================================================

bench.section("Pure Sine Waves - Round-trip (FFT + IFFT)");

for (const testCase of sineBinCentered.slice(0, 3)) {
  const input = Float64Array.from(testCase.signal);

  const result = bench.time(`round-trip (${testCase.name})`, () => {
    const forward = sineFft.forward(input);
    const inverse = sineFft.inverse(forward);
    return inverse;
  });

  // Verify round-trip accuracy (not timed)
  const error = maxAbsError(result.real, input);
  if (error > 1e-10) {
    console.warn(`  WARNING: Round-trip error ${error} exceeds tolerance`);
  }
}

bench.memory("After round-trip");

// =============================================================================
// Multi-Tone Signals
// =============================================================================

bench.section("Multi-Tone Signals");

const multiRef = loadMultiTone();
const multiFft = new FFT(multiRef.n);

for (const testCase of multiRef.cases) {
  const input = Float64Array.from(testCase.signal);

  bench.time(`FFT forward (${testCase.name}, n=${testCase.n})`, () => {
    return multiFft.forward(input);
  });

  // Verify peaks (not timed)
  const fftResult = multiFft.forward(input);
  const mag = magnitude(fftResult);
  const binIndices = testCase.params.bin_indices as number[];
  const amplitudes = testCase.params.amplitudes as number[];

  for (let i = 0; i < binIndices.length; i += 1) {
    const bin = binIndices[i]!;
    const expectedAmp = amplitudes[i]!;
    const expectedMag = (testCase.n * expectedAmp) / 2;
    const actualMag = mag[bin]!;
    const relError = Math.abs(actualMag - expectedMag) / expectedMag;
    if (relError > 0.01) {
      console.warn(
        `  WARNING: Peak at bin ${bin} has ${relError * 100}% error`
      );
    }
  }
}

bench.time(`round-trip (${multiRef.cases[0]!.name})`, () => {
  const input = Float64Array.from(multiRef.cases[0]!.signal);
  const forward = multiFft.forward(input);
  return multiFft.inverse(forward);
});

bench.memory("After multi-tone");

// =============================================================================
// Chirp Signals
// =============================================================================

bench.section("Chirp Signals");

const chirpRef = loadChirp();
const chirpFft = new FFT(chirpRef.n);

for (const testCase of chirpRef.cases) {
  const input = Float64Array.from(testCase.signal);

  bench.time(`FFT forward (${testCase.name}, n=${testCase.n})`, () => {
    return chirpFft.forward(input);
  });

  const result = bench.time(`round-trip (${testCase.name})`, () => {
    const forward = chirpFft.forward(input);
    return chirpFft.inverse(forward);
  });

  // Verify round-trip accuracy
  const error = maxAbsError(result.real, input);
  if (error > 1e-10) {
    console.warn(`  WARNING: Chirp round-trip error ${error} exceeds tolerance`);
  }
}

bench.memory("After chirp");

// =============================================================================
// Special Signals
// =============================================================================

bench.section("Special Signals");

const specialRef = loadSpecial();
const specialFft = new FFT(specialRef.n);

const impulseCase = specialRef.cases.find((c) => c.kind === "impulse");
if (impulseCase) {
  const input = Float64Array.from(impulseCase.signal);

  bench.time(`FFT forward (impulse, n=${impulseCase.n})`, () => {
    return specialFft.forward(input);
  });

  // Verify flat magnitude
  const fftResult = specialFft.forward(input);
  const mag = magnitude(fftResult);
  if (impulseCase.params.position === 0) {
    const expectedMag = impulseCase.params.amplitude as number;
    for (let i = 0; i < impulseCase.n; i += 1) {
      if (Math.abs(mag[i]! - expectedMag) > 1e-10) {
        console.warn(`  WARNING: Impulse magnitude not flat at bin ${i}`);
        break;
      }
    }
  }
}

const dcCase = specialRef.cases.find((c) => c.kind === "dc");
if (dcCase) {
  const input = Float64Array.from(dcCase.signal);

  bench.time(`FFT forward (DC, n=${dcCase.n})`, () => {
    return specialFft.forward(input);
  });

  // Verify DC only in bin 0
  const fftResult = specialFft.forward(input);
  const mag = magnitude(fftResult);
  const expectedDc = dcCase.n * (dcCase.params.level as number);
  if (Math.abs(mag[0]! - expectedDc) > 1e-10) {
    console.warn(`  WARNING: DC magnitude mismatch`);
  }
}

const nyquistCase = specialRef.cases.find((c) => c.kind === "nyquist");
if (nyquistCase) {
  const input = Float64Array.from(nyquistCase.signal);

  bench.time(`FFT forward (Nyquist, n=${nyquistCase.n})`, () => {
    return specialFft.forward(input);
  });

  // Verify energy only at Nyquist bin
  const fftResult = specialFft.forward(input);
  const mag = magnitude(fftResult);
  const nyquistBin = nyquistCase.n / 2;
  const expectedMag = nyquistCase.n * (nyquistCase.params.amplitude as number);
  if (Math.abs(mag[nyquistBin]! - expectedMag) > 1e-10) {
    console.warn(`  WARNING: Nyquist magnitude mismatch`);
  }
}

const zerosCase = specialRef.cases.find((c) => c.kind === "zeros");
if (zerosCase) {
  const input = Float64Array.from(zerosCase.signal);

  bench.time(`FFT forward (zeros, n=${zerosCase.n})`, () => {
    return specialFft.forward(input);
  });
}

bench.memory("After special signals");

// =============================================================================
// Batch Processing Simulation
// =============================================================================

bench.section("Batch Processing (100 frames)");

const batchInput = Float64Array.from(sineBinCentered[0]!.signal);
const batchFft = new FFT(sineRef.n);

bench.time(`100 FFT forward (n=${sineRef.n})`, () => {
  let lastResult;
  for (let i = 0; i < 100; i += 1) {
    lastResult = batchFft.forward(batchInput);
  }
  return lastResult;
}, { iterations: 10 });

bench.time(`100 FFT forward + magnitude (n=${sineRef.n})`, () => {
  let lastMag;
  for (let i = 0; i < 100; i += 1) {
    const fftResult = batchFft.forward(batchInput);
    lastMag = magnitude(fftResult);
  }
  return lastMag;
}, { iterations: 10 });

bench.time(`100 round-trips (n=${sineRef.n})`, () => {
  let lastResult;
  for (let i = 0; i < 100; i += 1) {
    const forward = batchFft.forward(batchInput);
    lastResult = batchFft.inverse(forward);
  }
  return lastResult;
}, { iterations: 10 });

bench.memory("After batch processing");

// =============================================================================
// Report
// =============================================================================

bench.report();
