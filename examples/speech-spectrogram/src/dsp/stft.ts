/**
 * Local STFT helper – loops over overlapping frames calling pragma-dsp's
 * FFT and collects magnitude/phase slices for spectrogram rendering.
 *
 * This is deliberately kept app-local to inform v0.2 STFT API design.
 */
import {
  FFT,
  createWindow,
  applyWindow,
  magnitude,
  phase,
  binFrequencies,
  type WindowType,
} from "pragma-dsp/xform/fourier";
import type { ComplexArray } from "pragma-dsp/core";

export type StftOptions = {
  fftSize?: number;
  hopSize?: number;
  window?: WindowType;
  sampleRate?: number;
};

export type StftFrame = {
  /** Raw complex FFT output (full N bins) */
  complex: ComplexArray;
  /** Magnitude spectrum (N/2+1 bins, one-sided) */
  magnitudes: Float64Array;
  /** Phase spectrum (N/2+1 bins, one-sided) */
  phases: Float64Array;
};

export type StftResult = {
  frames: StftFrame[];
  /** One-sided bin frequencies */
  frequencies: Float64Array;
  /** Time (in seconds) of the center of each frame */
  times: Float64Array;
  fftSize: number;
  hopSize: number;
  sampleRate: number;
};

/**
 * Compute a Short-Time Fourier Transform over `samples`.
 *
 * Since pragma-dsp v0.1 has no built-in STFT, this manually loops with
 * overlap, windows each frame, runs FFT, and collects results.
 */
export function computeStft(
  samples: Float32Array | Float64Array,
  opts: StftOptions = {}
): StftResult {
  const fftSize = opts.fftSize ?? 2048;
  const hopSize = opts.hopSize ?? Math.floor(fftSize / 4);
  const windowType = opts.window ?? "hann";
  const sampleRate = opts.sampleRate ?? 16_000;

  const fft = new FFT(fftSize);
  const win = createWindow(windowType, fftSize);
  const frequencies = binFrequencies(fftSize, sampleRate, "one");
  const binCount = frequencies.length; // N/2 + 1

  const frames: StftFrame[] = [];
  const timeCenters: number[] = [];

  const frameBuffer = new Float64Array(fftSize);

  for (let start = 0; start + fftSize <= samples.length; start += hopSize) {
    // Copy frame into Float64Array buffer
    for (let i = 0; i < fftSize; i++) {
      frameBuffer[i] = samples[start + i]!;
    }

    // Apply window
    const windowed = applyWindow(frameBuffer, win);

    // FFT forward → complex
    const complex = fft.forward(windowed);

    // Magnitude + phase (one-sided)
    const fullMag = magnitude(complex);
    const fullPhase = phase(complex);

    const magnitudes = new Float64Array(binCount);
    const phases = new Float64Array(binCount);
    for (let k = 0; k < binCount; k++) {
      magnitudes[k] = fullMag[k]!;
      phases[k] = fullPhase[k]!;
    }

    frames.push({ complex, magnitudes, phases });
    timeCenters.push((start + fftSize / 2) / sampleRate);
  }

  return {
    frames,
    frequencies,
    times: Float64Array.from(timeCenters),
    fftSize,
    hopSize,
    sampleRate,
  };
}

/**
 * Scale magnitudes into dB, clamped to a floor.
 */
export function magnitudeToDb(
  mag: Float64Array,
  floorDb: number = -80
): Float64Array {
  const db = new Float64Array(mag.length);
  for (let i = 0; i < mag.length; i++) {
    const val = mag[i]!;
    db[i] = val > 0 ? 20 * Math.log10(val) : floorDb;
    if (db[i]! < floorDb) db[i] = floorDb;
  }
  return db;
}
