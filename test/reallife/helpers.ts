import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect } from "vitest";

const refsDir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "references"
);

// =============================================================================
// Reference types
// =============================================================================

export type SignalCase = {
  name: string;
  kind: string;
  n: number;
  sampleRate: number;
  signal: number[];
  fftRe: number[];
  fftIm: number[];
  magnitude: number[];
  phase: number[];
  peakBin: number;
  peakMagnitude: number;
  peakPhase: number;
  params: Record<string, unknown>;
};

export type SignalReference = {
  generatedAt: string;
  generator: string;
  python: string;
  numpy: string;
  scipy: string;
  platform: string;
  description: string;
  n: number;
  sampleRate: number;
  cases: SignalCase[];
};

export type WindowDspCase = {
  type: "rect" | "hann" | "hamming" | "blackman";
  n: number;
  values: number[];
  coherentGain: number;
  enbw: number;
};

export type WindowDspReference = {
  generatedAt: string;
  generator: string;
  python: string;
  numpy: string;
  scipy: string;
  platform: string;
  description: string;
  cases: WindowDspCase[];
};

// =============================================================================
// Reference loaders
// =============================================================================

export const loadReference = <T>(name: string): T => {
  const path = resolve(refsDir, `${name}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as T;
};

export const loadPureSine = (): SignalReference =>
  loadReference<SignalReference>("pure_sine");

export const loadCosine = (): SignalReference =>
  loadReference<SignalReference>("cosine");

export const loadMultiTone = (): SignalReference =>
  loadReference<SignalReference>("multi_tone");

export const loadChirp = (): SignalReference =>
  loadReference<SignalReference>("chirp");

export const loadSpecial = (): SignalReference =>
  loadReference<SignalReference>("special");

export const loadWindowsDsp = (): WindowDspReference =>
  loadReference<WindowDspReference>("windows_dsp");

// =============================================================================
// Assertion helpers
// =============================================================================

/**
 * Assert that two arrays are element-wise close within tolerance.
 */
export const expectCloseArray = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tol = 1e-10
): void => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    const a = actual[i] ?? 0;
    const e = expected[i] ?? 0;
    const diff = Math.abs(a - e);
    if (diff > tol) {
      throw new Error(
        `Array mismatch at index ${i}: actual=${a}, expected=${e}, diff=${diff}, tol=${tol}`
      );
    }
  }
};

/**
 * Assert that two arrays are element-wise close within relative tolerance.
 */
export const expectCloseArrayRelative = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  relTol = 1e-6,
  absTol = 1e-12
): void => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    const a = actual[i] ?? 0;
    const e = expected[i] ?? 0;
    const diff = Math.abs(a - e);
    const tolerance = Math.max(absTol, relTol * Math.abs(e));
    if (diff > tolerance) {
      throw new Error(
        `Array mismatch at index ${i}: actual=${a}, expected=${e}, diff=${diff}, tolerance=${tolerance}`
      );
    }
  }
};

/**
 * Compute the maximum absolute error between two arrays.
 */
export const maxAbsError = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>
): number => {
  let max = 0;
  const len = Math.min(actual.length, expected.length);
  for (let i = 0; i < len; i += 1) {
    const diff = Math.abs((actual[i] ?? 0) - (expected[i] ?? 0));
    if (diff > max) max = diff;
  }
  return max;
};

/**
 * Compute the RMS error between two arrays.
 */
export const rmsError = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>
): number => {
  let sum = 0;
  const len = Math.min(actual.length, expected.length);
  for (let i = 0; i < len; i += 1) {
    const diff = (actual[i] ?? 0) - (expected[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum / len);
};

// =============================================================================
// Signal generation (TypeScript side for comparison)
// =============================================================================

/**
 * Generate a sine wave in TypeScript (for verifying signal generation matches).
 */
export const generateSine = (
  freqHz: number,
  amplitude: number,
  phaseRad: number,
  sampleRate: number,
  n: number
): Float64Array => {
  const signal = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const t = i / sampleRate;
    signal[i] = amplitude * Math.sin(2 * Math.PI * freqHz * t + phaseRad);
  }
  return signal;
};

/**
 * Generate a cosine wave in TypeScript.
 */
export const generateCosine = (
  freqHz: number,
  amplitude: number,
  phaseRad: number,
  sampleRate: number,
  n: number
): Float64Array => {
  const signal = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const t = i / sampleRate;
    signal[i] = amplitude * Math.cos(2 * Math.PI * freqHz * t + phaseRad);
  }
  return signal;
};

/**
 * Generate an impulse signal.
 */
export const generateImpulse = (
  n: number,
  position: number,
  amplitude: number
): Float64Array => {
  const signal = new Float64Array(n);
  signal[position] = amplitude;
  return signal;
};

/**
 * Generate a DC (constant) signal.
 */
export const generateDc = (n: number, level: number): Float64Array => {
  const signal = new Float64Array(n);
  signal.fill(level);
  return signal;
};

/**
 * Generate a Nyquist signal (alternating +1/-1).
 */
export const generateNyquist = (n: number, amplitude: number): Float64Array => {
  const signal = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    signal[i] = amplitude * (i % 2 === 0 ? 1 : -1);
  }
  return signal;
};
