// v0.1: FFT + windowing + helpers live here.
// This file should re-export the friendly Fourier transform surface.

import {
  type ComplexArray,
  Radix2Fft,
  createComplexArray,
  isPowerOfTwo
} from "../core/fft.js";

export type WindowType = "rect" | "hann" | "hamming" | "blackman";
export type FftSides = "one" | "two";

export const createWindow = (type: WindowType, size: number): Float64Array => {
  if (size <= 0) {
    throw new Error(`Window size must be positive, got ${size}`);
  }
  if (size === 1) {
    return new Float64Array([1]);
  }

  const out = new Float64Array(size);
  switch (type) {
    case "rect": {
      out.fill(1);
      return out;
    }
    case "hann": {
      for (let i = 0; i < size; i += 1) {
        out[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
      }
      return out;
    }
    case "hamming": {
      for (let i = 0; i < size; i += 1) {
        out[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (size - 1));
      }
      return out;
    }
    case "blackman": {
      for (let i = 0; i < size; i += 1) {
        const f = (2 * Math.PI * i) / (size - 1);
        out[i] = 0.42 - 0.5 * Math.cos(f) + 0.08 * Math.cos(2 * f);
      }
      return out;
    }
    default: {
      const exhaustive: never = type;
      throw new Error(`Unsupported window type: ${exhaustive}`);
    }
  }
};

export const applyWindow = (
  input: ArrayLike<number>,
  window: ArrayLike<number>,
  out?: Float64Array
): Float64Array => {
  if (input.length !== window.length) {
    throw new Error("Window length must match input length.");
  }
  const result = out ?? new Float64Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    result[i] = (input[i] ?? 0) * (window[i] ?? 0);
  }
  return result;
};

export class FFT {
  readonly size: number;
  private readonly kernel: Radix2Fft;

  constructor(size: number) {
    if (!isPowerOfTwo(size)) {
      throw new Error(`FFT size must be power of two, got ${size}`);
    }
    this.size = size;
    this.kernel = new Radix2Fft(size);
  }

  forward(input: ArrayLike<number>, out?: ComplexArray): ComplexArray {
    return this.kernel.forward(input, out);
  }

  forwardComplex(input: ComplexArray, out?: ComplexArray): ComplexArray {
    return this.kernel.forwardComplex(input, out);
  }

  inverse(input: ComplexArray, out?: ComplexArray): ComplexArray {
    return this.kernel.inverse(input, out);
  }

  createComplexArray(fill = 0): ComplexArray {
    return createComplexArray(this.size, fill);
  }
}

export const magnitude = (
  input: ComplexArray,
  out?: Float64Array
): Float64Array => {
  const result = out ?? new Float64Array(input.real.length);
  for (let i = 0; i < input.real.length; i += 1) {
    const re = input.real[i] ?? 0;
    const im = input.imag[i] ?? 0;
    result[i] = Math.hypot(re, im);
  }
  return result;
};

export const phase = (
  input: ComplexArray,
  out?: Float64Array
): Float64Array => {
  const result = out ?? new Float64Array(input.real.length);
  for (let i = 0; i < input.real.length; i += 1) {
    result[i] = Math.atan2(input.imag[i] ?? 0, input.real[i] ?? 0);
  }
  return result;
};

export const fftShift = (
  input: ArrayLike<number>,
  out?: Float64Array
): Float64Array => {
  const n = input.length;
  const result = out ?? new Float64Array(n);
  const mid = Math.floor(n / 2);
  for (let i = 0; i < n; i += 1) {
    const j = (i + mid) % n;
    result[i] = input[j] ?? 0;
  }
  return result;
};

export const fftShiftComplex = (
  input: ComplexArray,
  out?: ComplexArray
): ComplexArray => {
  const n = input.real.length;
  const result = out ?? createComplexArray(n);
  result.real.set(fftShift(input.real));
  result.imag.set(fftShift(input.imag));
  return result;
};

export const binFrequencies = (
  size: number,
  sampleRate: number,
  sides: FftSides = "one"
): Float64Array => {
  if (size <= 0) {
    throw new Error(`FFT size must be positive, got ${size}`);
  }
  if (sampleRate <= 0) {
    throw new Error(`Sample rate must be positive, got ${sampleRate}`);
  }
  const binCount = sides === "one" ? Math.floor(size / 2) + 1 : size;
  const result = new Float64Array(binCount);
  const scale = sampleRate / size;
  for (let i = 0; i < binCount; i += 1) {
    result[i] = i * scale;
  }
  return result;
};
