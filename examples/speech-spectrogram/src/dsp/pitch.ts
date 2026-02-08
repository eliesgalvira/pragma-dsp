/**
 * Pitch (F0) and formant detection using pragma-dsp.
 *
 * F0: autocorrelation via Wiener–Khinchin theorem
 *     signal → FFT → |·|² → IFFT → find first peak after lag 0
 *
 * Formants: smooth the spectral envelope with a moving average,
 *           then pick the first N local maxima → F1, F2, F3, ...
 */
import { FFT, magnitude } from "pragma-dsp/xform/fourier";
import type { ComplexArray } from "pragma-dsp/core";

// ── Helpers ──────────────────────────────────────────────────────────

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ── Pitch detection (autocorrelation via FFT) ────────────────────────

export type PitchResult = {
  f0: number | null; // Hz, null if unvoiced / no clear pitch
  confidence: number; // 0..1
};

/**
 * Estimate fundamental frequency via autocorrelation (Wiener–Khinchin):
 *   R(τ) = IFFT( |FFT(x)|² )
 *
 * Then find the first prominent peak in R after a minimum lag.
 */
export function detectPitch(
  samples: Float32Array | Float64Array,
  sampleRate: number,
  opts: { minF0?: number; maxF0?: number } = {}
): PitchResult {
  const minF0 = opts.minF0 ?? 50; // Hz
  const maxF0 = opts.maxF0 ?? 600; // Hz

  const N = nextPow2(samples.length * 2); // zero-pad for linear autocorrelation
  const fft = new FFT(N);

  // Forward FFT of zero-padded signal
  const input = new Float64Array(N);
  for (let i = 0; i < samples.length; i++) input[i] = samples[i]!;
  const spectrum = fft.forward(input);

  // Power spectrum: |X[k]|²  → stored back in a ComplexArray (real part only)
  const power: ComplexArray = { real: new Float64Array(N), imag: new Float64Array(N) };
  for (let k = 0; k < N; k++) {
    const re = spectrum.real[k]!;
    const im = spectrum.imag[k]!;
    power.real[k] = re * re + im * im;
    // imag stays 0
  }

  // IFFT → autocorrelation
  const autocorr = fft.inverse(power);
  const r = autocorr.real;

  // r[0] is the signal energy (autocorrelation at lag 0)
  const r0 = r[0]!;
  if (r0 === 0) return { f0: null, confidence: 0 };

  // Search for first peak in valid lag range
  const minLag = Math.floor(sampleRate / maxF0);
  const maxLag = Math.min(Math.ceil(sampleRate / minF0), samples.length - 1);

  let bestLag = minLag;
  let bestVal = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    const val = r[lag]!;
    if (val > bestVal) {
      bestVal = val;
      bestLag = lag;
    }
  }

  // Parabolic interpolation around peak for sub-sample accuracy
  const prev = bestLag > 0 ? r[bestLag - 1]! : bestVal;
  const next = bestLag < r.length - 1 ? r[bestLag + 1]! : bestVal;
  const shift = (prev - next) / (2 * (prev - 2 * bestVal + next) || 1);
  const refinedLag = bestLag + shift;

  const confidence = bestVal / r0;
  const f0 = confidence > 0.2 ? sampleRate / refinedLag : null;

  return { f0, confidence };
}

// ── Formant detection (spectral envelope peak-picking) ───────────────

export type FormantResult = {
  /** Formant frequencies in Hz (F1, F2, F3, ...) */
  formants: number[];
  /** Smoothed spectral envelope (magnitudes, one-sided) */
  envelope: Float64Array;
};

/**
 * Estimate formants by smoothing the magnitude spectrum and picking peaks.
 *
 * The approach is intentionally simple (moving-average envelope → local maxima)
 * rather than LPC, to exercise pragma-dsp's spectrum helpers.
 */
export function detectFormants(
  magnitudes: Float64Array,
  frequencies: Float64Array,
  opts: { smoothingWidth?: number; maxFormants?: number; minFreq?: number; maxFreq?: number } = {}
): FormantResult {
  const smoothingWidth = opts.smoothingWidth ?? 11;
  const maxFormants = opts.maxFormants ?? 4;
  const minFreq = opts.minFreq ?? 200;
  const maxFreq = opts.maxFreq ?? 5500;

  const N = magnitudes.length;

  // ── Smooth with moving average ────
  const half = Math.floor(smoothingWidth / 2);
  const envelope = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(N - 1, i + half); j++) {
      sum += magnitudes[j]!;
      count++;
    }
    envelope[i] = sum / count;
  }

  // ── Pick local maxima within freq range ────
  const peaks: { freq: number; mag: number }[] = [];

  for (let i = 1; i < N - 1; i++) {
    const freq = frequencies[i]!;
    if (freq < minFreq || freq > maxFreq) continue;

    const prev = envelope[i - 1]!;
    const curr = envelope[i]!;
    const next = envelope[i + 1]!;

    if (curr > prev && curr > next && curr > 0) {
      peaks.push({ freq, mag: curr });
    }
  }

  // Sort by magnitude descending, take top N, then sort by frequency
  peaks.sort((a, b) => b.mag - a.mag);
  const top = peaks.slice(0, maxFormants);
  top.sort((a, b) => a.freq - b.freq);

  return {
    formants: top.map((p) => p.freq),
    envelope,
  };
}
