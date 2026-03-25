import type { ComplexArray } from "../core/fft.js";
import { createComplexArray } from "../core/fft.js";
import {
  FFT,
  applyWindow,
  binFrequencies,
  createWindow,
  type FftSides,
  type WindowType,
} from "./fourier.js";

export type StftWindow = WindowType | ArrayLike<number>;

export type StftOptions = {
  fftSize?: number;
  hopSize?: number;
  window?: StftWindow;
  sampleRate?: number;
  sides?: FftSides;
  /**
   * One-sided complex storage for real signals.
   * This is a storage-only optimization; the inverse expands via conjugate symmetry.
   */
  complexSides?: "one" | "two";
};

export type StftFrame = {
  /** Raw complex FFT output (full or one-sided based on `complexSides`). */
  complex: ComplexArray;
  /** Magnitude spectrum (one-sided if sides === "one"). */
  magnitudes: Float64Array;
  /** Phase spectrum (one-sided if sides === "one"). */
  phases: Float64Array;
};

export type StftResult = {
  frames: StftFrame[];
  /** Bin frequencies for `magnitudes`/`phases`. */
  frequencies: Float64Array;
  /** Time (seconds) at the center of each frame. */
  times: Float64Array;
  fftSize: number;
  hopSize: number;
  sampleRate: number;
  sides: FftSides;
  /** Storage mode for complex frames. */
  complexSides: "one" | "two";
  /** Analysis window used for each frame. */
  window: Float64Array;
  /**
   * Inverse STFT (overlap-add).
   * Uses the same window/hop/fftSize and expands one-sided complex frames.
   */
  inverse: (options?: { outputLength?: number }) => Float64Array;
};

export type SpectrogramResult = {
  readonly stft: StftResult;
  readonly dbFrames: ReadonlyArray<Float64Array>;
  readonly floorDb: number;
};

export type SpectrogramOptions = StftOptions & {
  readonly floorDb?: number;
};

export type IstftOptions = {
  fftSize: number;
  hopSize: number;
  window: StftWindow | Float64Array;
  complexSides: "one" | "two";
  outputLength?: number;
};

const defaultFftSize = (samplesLength: number): number => {
  if (samplesLength <= 0) {
    throw new Error("STFT requires a non-empty input signal.");
  }
  const power = Math.floor(Math.log2(samplesLength));
  const size = 1 << Math.max(0, power);
  return Math.max(1, size);
};

const resolveWindow = (
  windowSpec: StftWindow | undefined,
  fftSize: number
): Float64Array => {
  if (windowSpec === undefined) {
    return createWindow("hann", fftSize);
  }
  if (typeof windowSpec === "string") {
    return createWindow(windowSpec, fftSize);
  }
  return Float64Array.from(windowSpec);
};

/**
 * Expand one-sided complex bins (0..Nyquist) into a full complex spectrum.
 * Assumes real-signal conjugate symmetry.
 */
export const expandOneSidedComplex = (
  oneSided: ComplexArray,
  fftSize: number
): ComplexArray => {
  const half = Math.floor(fftSize / 2);
  if (oneSided.real.length !== half + 1) {
    throw new Error("One-sided complex length must be N/2 + 1.");
  }

  const full = createComplexArray(fftSize);
  full.real.set(oneSided.real);
  full.imag.set(oneSided.imag);

  for (let k = 1; k < half; k += 1) {
    full.real[fftSize - k] = oneSided.real[k] ?? 0;
    full.imag[fftSize - k] = -(oneSided.imag[k] ?? 0);
  }

  return full;
};

export const istft = (
  frames: StftFrame[],
  options: IstftOptions
): Float64Array => {
  const { fftSize, hopSize, window, complexSides, outputLength } = options;
  if (fftSize <= 0) {
    throw new Error(`FFT size must be positive, got ${fftSize}`);
  }
  if (hopSize <= 0) {
    throw new Error(`Hop size must be positive, got ${hopSize}`);
  }

  const windowArray =
    window instanceof Float64Array
      ? window
      : resolveWindow(window, fftSize);

  if (windowArray.length !== fftSize) {
    throw new Error("Window length must match FFT size.");
  }

  const frameCount = frames.length;
  const length = Math.max(0, (frameCount - 1) * hopSize + fftSize);
  const output = new Float64Array(length);
  const denom = new Float64Array(length);
  const fft = new FFT(fftSize);

  for (let f = 0; f < frameCount; f += 1) {
    const frame = frames[f]!;
    const complex =
      complexSides === "one"
        ? expandOneSidedComplex(frame.complex, fftSize)
        : frame.complex;
    const time = fft.inverse(complex);
    const start = f * hopSize;
    for (let i = 0; i < fftSize; i += 1) {
      const idx = start + i;
      if (idx >= output.length) break;
      const w = windowArray[i] ?? 0;
      output[idx] = (output[idx] ?? 0) + (time.real[i] ?? 0) * w;
      denom[idx] = (denom[idx] ?? 0) + w * w;
    }
  }

  for (let i = 0; i < output.length; i += 1) {
    const d = denom[i] ?? 0;
    if (d > 0) {
      output[i] = (output[i] ?? 0) / d;
    }
  }

  return outputLength !== undefined ? output.slice(0, outputLength) : output;
};

/**
 * Short-time Fourier transform over a signal.
 * - `complexSides: "one"` is a storage-only optimization for real signals.
 * - The inverse expands via conjugate symmetry internally.
 */
export const stft = (
  samples: ArrayLike<number>,
  options: StftOptions = {}
): StftResult => {
  const sampleRate = options.sampleRate ?? 1;
  if (sampleRate <= 0) {
    throw new Error(`Sample rate must be positive, got ${sampleRate}`);
  }

  const windowSpec = options.window;
  const windowIsArray = windowSpec !== undefined && typeof windowSpec !== "string";
  const fftSize =
    options.fftSize ??
    (windowIsArray ? windowSpec.length : defaultFftSize(samples.length));
  if (fftSize <= 0) {
    throw new Error(`FFT size must be positive, got ${fftSize}`);
  }

  const hopSize = options.hopSize ?? Math.floor(fftSize / 4);
  if (hopSize <= 0) {
    throw new Error(`Hop size must be positive, got ${hopSize}`);
  }

  const sides: FftSides = options.sides ?? "one";
  const complexSides = options.complexSides ?? "two";
  if (complexSides === "one" && sides === "two") {
    throw new Error("complexSides:\"one\" requires sides:\"one\".");
  }

  const window = resolveWindow(windowSpec, fftSize);

  if (window.length !== fftSize) {
    throw new Error("Window length must match FFT size.");
  }

  const fft = new FFT(fftSize);
  const frequencies = binFrequencies(fftSize, sampleRate, sides);
  const binCount = frequencies.length;

  const frames: StftFrame[] = [];
  const timeCenters: number[] = [];
  const frameBuffer = new Float64Array(fftSize);
  const windowed = new Float64Array(fftSize);

  for (let start = 0; start + fftSize <= samples.length; start += hopSize) {
    for (let i = 0; i < fftSize; i += 1) {
      frameBuffer[i] = samples[start + i] ?? 0;
    }

    applyWindow(frameBuffer, window, windowed);

    const complexFull = fft.createComplexArray();
    fft.forward(windowed, complexFull);

    const magnitudes = new Float64Array(binCount);
    const phases = new Float64Array(binCount);
    for (let k = 0; k < binCount; k += 1) {
      const re = complexFull.real[k] ?? 0;
      const im = complexFull.imag[k] ?? 0;
      magnitudes[k] = Math.hypot(re, im);
      phases[k] = Math.atan2(im, re);
    }

    const complex =
      complexSides === "one"
        ? {
            real: complexFull.real.slice(0, binCount),
            imag: complexFull.imag.slice(0, binCount),
          }
        : complexFull;

    frames.push({ complex, magnitudes, phases });
    timeCenters.push((start + fftSize / 2) / sampleRate);
  }

  const result: StftResult = {
    frames,
    frequencies,
    times: Float64Array.from(timeCenters),
    fftSize,
    hopSize,
    sampleRate,
    sides,
    complexSides,
    window,
    inverse: (opts = {}) =>
      istft(
        frames,
        opts.outputLength === undefined
          ? {
              fftSize,
              hopSize,
              window,
              complexSides,
            }
          : {
              fftSize,
              hopSize,
              window,
              complexSides,
              outputLength: opts.outputLength,
            }
      ),
  };
  return result;
};

export function spectrogram(
  signal: ArrayLike<number>,
  options?: SpectrogramOptions,
): SpectrogramResult;
export function spectrogram(
  signal: StftResult,
  options?: Pick<SpectrogramOptions, "floorDb">,
): SpectrogramResult;
export function spectrogram(
  signal: ArrayLike<number> | StftResult,
  options: SpectrogramOptions | Pick<SpectrogramOptions, "floorDb"> = {},
): SpectrogramResult {
  const stftResult =
    "frames" in signal && "frequencies" in signal && "times" in signal
      ? signal
      : stft(signal, {
          fftSize: (options as SpectrogramOptions).fftSize,
          hopSize: (options as SpectrogramOptions).hopSize,
          window: (options as SpectrogramOptions).window,
          sampleRate: (options as SpectrogramOptions).sampleRate,
          sides: (options as SpectrogramOptions).sides,
          complexSides: (options as SpectrogramOptions).complexSides,
        });
  const floorDb = options.floorDb ?? -80;

  const dbFrames = stftResult.frames.map((frame) => {
    const values = new Float64Array(frame.magnitudes.length);
    for (let index = 0; index < frame.magnitudes.length; index++) {
      const magnitude = frame.magnitudes[index] ?? 0;
      const db = magnitude > 0 ? 20 * Math.log10(magnitude) : floorDb;
      values[index] = db < floorDb ? floorDb : db;
    }
    return values;
  });

  return {
    stft: stftResult,
    dbFrames,
    floorDb,
  };
}
