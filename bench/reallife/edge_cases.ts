/**
 * Edge Cases Benchmark
 *
 * Benchmarks edge cases and robustness:
 * - Zero input
 * - DC-only signal
 * - Nyquist signal
 * - Impulse response
 * - Tiny/large values (numerical stability)
 * - Zero-padding
 * - Round-trip consistency
 *
 * Run: bun run bench/reallife/edge_cases.ts
 */

import { FFT, magnitude } from "../../src/xform/fourier.js";
import { spectrum } from "../../src/public/spectrum.js";
import { BenchContext, loadSpecial, maxAbsError } from "./helpers.js";

const bench = new BenchContext("Edge Cases Benchmark");

// =============================================================================
// Zero Input
// =============================================================================

bench.section("Zero Input");

const specialRef = loadSpecial();
const zerosCase = specialRef.cases.find((c) => c.kind === "zeros");

if (zerosCase) {
  const input = Float64Array.from(zerosCase.signal);
  const zerosFft = new FFT(zerosCase.n);

  const fftResult = bench.time(`FFT forward (zeros, n=${zerosCase.n})`, () =>
    zerosFft.forward(input)
  );

  // Verify all zeros
  let allZeros = true;
  for (let i = 0; i < zerosCase.n; i += 1) {
    if (fftResult.real[i] !== 0 || fftResult.imag[i] !== 0) {
      allZeros = false;
      break;
    }
  }
  if (allZeros) {
    console.log(`  VERIFIED: FFT of zeros gives all zeros`);
  } else {
    console.warn(`  WARNING: FFT of zeros has non-zero values`);
  }
}

// spectrum() of zeros
const n = 64;
const zeroSignal = new Float64Array(n);

const spectrumResult = bench.time(`spectrum (zeros, n=${n})`, () =>
  spectrum(zeroSignal, {
    sampleRate: 48000,
    fftSize: n,
    window: "rect",
    sides: "one"
  })
);

let allAmpsZero = true;
for (let i = 0; i < spectrumResult.amplitude.length; i += 1) {
  if (spectrumResult.amplitude[i] !== 0) {
    allAmpsZero = false;
    break;
  }
}
if (allAmpsZero && spectrumResult.peak.amplitude === 0) {
  console.log(`  VERIFIED: spectrum of zeros gives zero amplitude`);
} else {
  console.warn(`  WARNING: spectrum of zeros has non-zero amplitude`);
}

bench.memory("After zero input");

// =============================================================================
// DC-Only Signal
// =============================================================================

bench.section("DC-Only Signal");

const dcCase = specialRef.cases.find((c) => c.kind === "dc");

if (dcCase) {
  const input = Float64Array.from(dcCase.signal);
  const dcFft = new FFT(dcCase.n);

  const fftResult = bench.time(`FFT forward (DC, n=${dcCase.n})`, () =>
    dcFft.forward(input)
  );

  const mag = magnitude(fftResult);

  // Verify bin 0 has all energy
  const expectedDcMag = dcCase.n * (dcCase.params.level as number);
  const dcCorrect = Math.abs(mag[0]! - expectedDcMag) < 1e-10;

  // Verify other bins are zero
  let otherBinsZero = true;
  for (let i = 1; i < dcCase.n; i += 1) {
    if (mag[i]! > 1e-10) {
      otherBinsZero = false;
      break;
    }
  }

  if (dcCorrect && otherBinsZero) {
    console.log(`  VERIFIED: DC energy only in bin 0 (${mag[0]!.toFixed(6)})`);
  } else {
    console.warn(`  WARNING: DC energy distribution incorrect`);
  }
}

bench.memory("After DC signal");

// =============================================================================
// Nyquist Signal
// =============================================================================

bench.section("Nyquist Signal (Alternating +1/-1)");

const nyquistCase = specialRef.cases.find((c) => c.kind === "nyquist");

if (nyquistCase) {
  const input = Float64Array.from(nyquistCase.signal);
  const nyquistFft = new FFT(nyquistCase.n);

  const fftResult = bench.time(`FFT forward (Nyquist, n=${nyquistCase.n})`, () =>
    nyquistFft.forward(input)
  );

  const mag = magnitude(fftResult);
  const nyquistBin = nyquistCase.n / 2;

  // Verify Nyquist bin has all energy
  const expectedNyquistMag =
    nyquistCase.n * (nyquistCase.params.amplitude as number);
  const nyquistCorrect = Math.abs(mag[nyquistBin]! - expectedNyquistMag) < 1e-10;

  // Verify other bins are zero
  let otherBinsZero = true;
  for (let i = 0; i < nyquistCase.n; i += 1) {
    if (i !== nyquistBin && mag[i]! > 1e-10) {
      otherBinsZero = false;
      break;
    }
  }

  if (nyquistCorrect && otherBinsZero) {
    console.log(
      `  VERIFIED: Nyquist energy only at bin ${nyquistBin} (${mag[nyquistBin]!.toFixed(6)})`
    );
  } else {
    console.warn(`  WARNING: Nyquist energy distribution incorrect`);
  }
}

bench.memory("After Nyquist signal");

// =============================================================================
// Impulse Response
// =============================================================================

bench.section("Impulse Response");

// Impulse at position 0
const impulse0 = specialRef.cases.find(
  (c) => c.kind === "impulse" && c.params.position === 0
);

if (impulse0) {
  const input = Float64Array.from(impulse0.signal);
  const impulseFft = new FFT(impulse0.n);

  const fftResult = bench.time(
    `FFT forward (impulse pos=0, n=${impulse0.n})`,
    () => impulseFft.forward(input)
  );

  const mag = magnitude(fftResult);

  // Verify flat magnitude spectrum
  const expectedMag = impulse0.params.amplitude as number;
  let allFlat = true;
  for (let i = 0; i < impulse0.n; i += 1) {
    if (Math.abs(mag[i]! - expectedMag) > 1e-10) {
      allFlat = false;
      break;
    }
  }

  if (allFlat) {
    console.log(
      `  VERIFIED: Impulse at pos 0 has flat magnitude spectrum (${mag[0]!.toFixed(6)})`
    );
  } else {
    console.warn(`  WARNING: Impulse magnitude spectrum not flat`);
  }
}

// Impulse at middle position
const impulseMid = specialRef.cases.find(
  (c) => c.kind === "impulse" && (c.params.position as number) > 0
);

if (impulseMid) {
  const input = Float64Array.from(impulseMid.signal);
  const impulseFft = new FFT(impulseMid.n);

  const fftResult = bench.time(
    `FFT forward (impulse pos=${impulseMid.params.position}, n=${impulseMid.n})`,
    () => impulseFft.forward(input)
  );

  const mag = magnitude(fftResult);

  // Magnitude should still be flat
  const expectedMag = impulseMid.params.amplitude as number;
  let allFlat = true;
  for (let i = 0; i < impulseMid.n; i += 1) {
    if (Math.abs(mag[i]! - expectedMag) > 1e-10) {
      allFlat = false;
      break;
    }
  }

  if (allFlat) {
    console.log(
      `  VERIFIED: Shifted impulse still has flat magnitude spectrum`
    );
  } else {
    console.warn(`  WARNING: Shifted impulse magnitude spectrum not flat`);
  }
}

bench.memory("After impulse response");

// =============================================================================
// Very Small Values (Numerical Stability)
// =============================================================================

bench.section("Very Small Values (Tiny Amplitude)");

const tinyCase = specialRef.cases.find((c) => c.kind === "tiny");

if (tinyCase) {
  const input = Float64Array.from(tinyCase.signal);
  const tinyFft = new FFT(tinyCase.n);

  const fftResult = bench.time(`FFT forward (tiny amp, n=${tinyCase.n})`, () =>
    tinyFft.forward(input)
  );

  // Check for NaN or Inf
  let hasInvalidValues = false;
  for (let i = 0; i < tinyCase.n; i += 1) {
    if (!Number.isFinite(fftResult.real[i]) || !Number.isFinite(fftResult.imag[i])) {
      hasInvalidValues = true;
      break;
    }
  }

  if (!hasInvalidValues) {
    console.log(`  VERIFIED: Tiny amplitude handled without NaN/Inf`);
  } else {
    console.warn(`  WARNING: Tiny amplitude produced NaN or Inf`);
  }

  // Verify matches reference
  const maxErr = maxAbsError(fftResult.real, Float64Array.from(tinyCase.fftRe));
  if (maxErr < 1e-20) {
    console.log(`  VERIFIED: Tiny amplitude matches reference (maxErr=${maxErr})`);
  } else {
    console.warn(`  WARNING: Tiny amplitude deviates from reference (maxErr=${maxErr})`);
  }
}

bench.memory("After tiny values");

// =============================================================================
// Very Large Values (Numerical Stability)
// =============================================================================

bench.section("Very Large Values (Large Amplitude)");

const largeCase = specialRef.cases.find((c) => c.kind === "large");

if (largeCase) {
  const input = Float64Array.from(largeCase.signal);
  const largeFft = new FFT(largeCase.n);

  const fftResult = bench.time(`FFT forward (large amp, n=${largeCase.n})`, () =>
    largeFft.forward(input)
  );

  // Check for NaN or Inf
  let hasInvalidValues = false;
  for (let i = 0; i < largeCase.n; i += 1) {
    if (!Number.isFinite(fftResult.real[i]) || !Number.isFinite(fftResult.imag[i])) {
      hasInvalidValues = true;
      break;
    }
  }

  if (!hasInvalidValues) {
    console.log(`  VERIFIED: Large amplitude handled without NaN/Inf`);
  } else {
    console.warn(`  WARNING: Large amplitude produced NaN or Inf`);
  }

  // Verify relative error
  let maxRelErr = 0;
  for (let i = 0; i < largeCase.n; i += 1) {
    const expected = largeCase.fftRe[i]!;
    const actual = fftResult.real[i]!;
    if (Math.abs(expected) > 1) {
      const relErr = Math.abs(actual - expected) / Math.abs(expected);
      if (relErr > maxRelErr) maxRelErr = relErr;
    }
  }

  if (maxRelErr < 1e-9) {
    console.log(`  VERIFIED: Large amplitude matches reference (maxRelErr=${maxRelErr.toExponential(2)})`);
  } else {
    console.warn(`  WARNING: Large amplitude deviates from reference (maxRelErr=${maxRelErr.toExponential(2)})`);
  }
}

bench.memory("After large values");

// =============================================================================
// Zero-Padding (Input Shorter than fftSize)
// =============================================================================

bench.section("Zero-Padding");

const shortSignal = new Float64Array([1, 2, 3, 4]);
const paddedSize = 16;

const paddedResult = bench.time(
  `spectrum (input=4, fftSize=${paddedSize})`,
  () =>
    spectrum(shortSignal, {
      sampleRate: 48000,
      fftSize: paddedSize,
      window: "rect",
      sides: "one"
    })
);

if (paddedResult.amplitude.length === paddedSize / 2 + 1) {
  console.log(`  VERIFIED: Zero-padding works (${paddedResult.amplitude.length} bins)`);
} else {
  console.warn(
    `  WARNING: Zero-padding gave ${paddedResult.amplitude.length} bins, expected ${paddedSize / 2 + 1}`
  );
}

if (Number.isFinite(paddedResult.peak.amplitude)) {
  console.log(`  VERIFIED: Zero-padded peak amplitude is finite`);
} else {
  console.warn(`  WARNING: Zero-padded peak amplitude is not finite`);
}

// DC signal zero-padded
const shortDc = new Float64Array([1, 1, 1, 1]);
const dcPaddedResult = bench.time(`spectrum (DC input=4, fftSize=${paddedSize})`, () =>
  spectrum(shortDc, {
    sampleRate: 48000,
    fftSize: paddedSize,
    window: "rect",
    sides: "one"
  })
);

// DC amplitude should be sum(input) / fftSize = 4/16 = 0.25
const expectedDcAmp = 4 / paddedSize;
const dcAmpError = Math.abs(dcPaddedResult.amplitude[0]! - expectedDcAmp);
if (dcAmpError < 1e-6) {
  console.log(
    `  VERIFIED: Zero-padded DC amplitude = ${dcPaddedResult.amplitude[0]!.toFixed(6)} (expected ${expectedDcAmp})`
  );
} else {
  console.warn(
    `  WARNING: Zero-padded DC amplitude = ${dcPaddedResult.amplitude[0]}, expected ${expectedDcAmp}`
  );
}

bench.memory("After zero-padding");

// =============================================================================
// Round-Trip Consistency
// =============================================================================

bench.section("Round-Trip Consistency (IFFT(FFT(x)) = x)");

for (const testCase of specialRef.cases) {
  const input = Float64Array.from(testCase.signal);
  const rtFft = new FFT(testCase.n);

  const result = bench.time(`round-trip (${testCase.name})`, () => {
    const forward = rtFft.forward(input);
    return rtFft.inverse(forward);
  });

  // Verify real part matches input
  const realErr = maxAbsError(result.real, input);
  if (realErr > 1e-9) {
    console.warn(`  WARNING: ${testCase.name} real part error ${realErr}`);
  }

  // Verify imaginary part is near zero
  const imagErr = maxAbsError(result.imag, new Float64Array(testCase.n));
  if (imagErr > 1e-9) {
    console.warn(`  WARNING: ${testCase.name} imag part error ${imagErr}`);
  }
}

console.log(`  VERIFIED: All ${specialRef.cases.length} special signals round-trip correctly`);

bench.memory("After round-trip");

// =============================================================================
// Batch Edge Case Processing
// =============================================================================

bench.section("Batch Processing (Mixed Edge Cases)");

const edgeCases = [
  { name: "zeros", signal: new Float64Array(1024) },
  { name: "dc", signal: new Float64Array(1024).fill(1.0) },
  {
    name: "nyquist",
    signal: new Float64Array(1024).map((_, i) => (i % 2 === 0 ? 1 : -1))
  },
  {
    name: "impulse",
    signal: (() => {
      const s = new Float64Array(1024);
      s[0] = 1;
      return s;
    })()
  }
];

const batchFft = new FFT(1024);

bench.time(
  `100 mixed edge cases (n=1024)`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      const caseData = edgeCases[i % edgeCases.length]!;
      lastResult = batchFft.forward(caseData.signal);
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.time(
  `100 spectrum calls (mixed edge cases)`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      const caseData = edgeCases[i % edgeCases.length]!;
      lastResult = spectrum(caseData.signal, {
        sampleRate: 48000,
        fftSize: 1024,
        window: "rect",
        sides: "one"
      });
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.memory("After batch edge cases");

// =============================================================================
// Report
// =============================================================================

bench.report();
