// v0.1: beginner-first API lives here.

import {
  FFT,
  type FftSides,
  type WindowType,
  applyWindow,
  binFrequencies,
  createWindow,
  magnitude,
  phase
} from "../xform/fourier.js";
import { nextPowerOfTwo } from "../core/fft.js";

export type SpectrumPeak = {
  index: number;
  frequency: number;
  amplitude: number;
  phase: number;
};

export type SpectrumResult = {
  frequencies: Float64Array;
  amplitude: Float64Array;
  phase: Float64Array;
  peak: SpectrumPeak;
};

export type SpectrumOptions = {
  sampleRate?: number;
  fftSize?: number;
  window?: WindowType;
  sides?: FftSides;
};

const buildFrame = (input: ArrayLike<number>, size: number): Float64Array => {
  const frame = new Float64Array(size);
  const limit = Math.min(size, input.length);
  for (let i = 0; i < limit; i += 1) {
    frame[i] = input[i] ?? 0;
  }
  return frame;
};

const scaleAmplitudeOneSided = (
  magnitudes: Float64Array,
  size: number
): Float64Array => {
  const binCount = Math.floor(size / 2) + 1;
  const result = new Float64Array(binCount);
  const nyquist = size % 2 === 0 ? size / 2 : -1;
  for (let k = 0; k < binCount; k += 1) {
    const mag = magnitudes[k] ?? 0;
    if (k === 0 || k === nyquist) {
      result[k] = mag / size;
    } else {
      result[k] = (2 * mag) / size;
    }
  }
  return result;
};

const scaleAmplitudeTwoSided = (
  magnitudes: Float64Array,
  size: number
): Float64Array => {
  const result = new Float64Array(size);
  for (let k = 0; k < size; k += 1) {
    result[k] = (magnitudes[k] ?? 0) / size;
  }
  return result;
};

const findPeak = (
  amplitude: Float64Array,
  frequencies: Float64Array
): SpectrumPeak => {
  let maxIndex = 0;
  let maxValue = amplitude[0] ?? 0;
  let hasNonDc = false;
  let nonDcIndex = 0;
  let nonDcValue = 0;
  for (let i = 1; i < amplitude.length; i += 1) {
    const v = amplitude[i] ?? 0;
    if (v > nonDcValue) {
      nonDcValue = v;
      nonDcIndex = i;
    }
    if (v > 0) {
      hasNonDc = true;
    }
    if (v > maxValue) {
      maxValue = v;
      maxIndex = i;
    }
  }

  const index = hasNonDc ? nonDcIndex : maxIndex;
  return {
    index,
    frequency: frequencies[index] ?? 0,
    amplitude: amplitude[index] ?? 0,
    phase: 0
  };
};

export const spectrum = (
  samples: ArrayLike<number>,
  options: SpectrumOptions = {}
): SpectrumResult => {
  const sampleRate = options.sampleRate ?? 1;
  const sides: FftSides = options.sides ?? "one";
  const targetSize = options.fftSize ?? nextPowerOfTwo(samples.length);
  const fft = new FFT(targetSize);
  const windowType: WindowType = options.window ?? "rect";
  const window = createWindow(windowType, targetSize);

  const frame = buildFrame(samples, targetSize);
  const windowed = applyWindow(frame, window);
  const spectrumComplex = fft.forward(windowed);
  const mag = magnitude(spectrumComplex);
  const ang = phase(spectrumComplex);

  const amplitude =
    sides === "one"
      ? scaleAmplitudeOneSided(mag, targetSize)
      : scaleAmplitudeTwoSided(mag, targetSize);
  const phaseBins =
    sides === "one"
      ? ang.slice(0, Math.floor(targetSize / 2) + 1)
      : ang;
  const frequencies = binFrequencies(targetSize, sampleRate, sides);
  const peak = findPeak(amplitude, frequencies);
  peak.phase = phaseBins[peak.index] ?? 0;

  return {
    frequencies,
    amplitude,
    phase: phaseBins,
    peak
  };
};
