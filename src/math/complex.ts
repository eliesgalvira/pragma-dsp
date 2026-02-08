/**
 * Pure, tree-shakeable complex-vector arithmetic on split-array representation.
 *
 * All functions operate on `ComplexArray` (`{ real, imag }` with Float64Array).
 * Each op has two forms:
 *   - allocating:  `scale(a, s)` → new ComplexArray
 *   - in-place:    `scaleInto(a, s, out)` → writes into `out`, returns `out`
 *
 * @module
 */

import type { ComplexArray } from "../core/fft.js";

// ── helpers ──────────────────────────────────────────────────────────

const len = (a: ComplexArray): number => a.real.length;

const alloc = (n: number): ComplexArray => ({
  real: new Float64Array(n),
  imag: new Float64Array(n),
});

// ── scale (scalar multiply) ──────────────────────────────────────────

/** Multiply every element by a real scalar. Returns a new ComplexArray. */
export const scale = (a: ComplexArray, s: number): ComplexArray =>
  scaleInto(a, s, alloc(len(a)));

/** Multiply every element by a real scalar, writing into `out`. */
export const scaleInto = (
  a: ComplexArray,
  s: number,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    out.real[i] = a.real[i]! * s;
    out.imag[i] = a.imag[i]! * s;
  }
  return out;
};

// ── add ──────────────────────────────────────────────────────────────

/** Element-wise complex addition. */
export const add = (a: ComplexArray, b: ComplexArray): ComplexArray =>
  addInto(a, b, alloc(len(a)));

export const addInto = (
  a: ComplexArray,
  b: ComplexArray,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    out.real[i] = a.real[i]! + b.real[i]!;
    out.imag[i] = a.imag[i]! + b.imag[i]!;
  }
  return out;
};

// ── sub ──────────────────────────────────────────────────────────────

/** Element-wise complex subtraction. */
export const sub = (a: ComplexArray, b: ComplexArray): ComplexArray =>
  subInto(a, b, alloc(len(a)));

export const subInto = (
  a: ComplexArray,
  b: ComplexArray,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    out.real[i] = a.real[i]! - b.real[i]!;
    out.imag[i] = a.imag[i]! - b.imag[i]!;
  }
  return out;
};

// ── mul (complex-by-complex, element-wise / Hadamard) ────────────────

/**
 * Element-wise complex multiply (Hadamard product).
 * `(a + ib)(c + id) = (ac − bd) + i(ad + bc)`
 */
export const mul = (a: ComplexArray, b: ComplexArray): ComplexArray =>
  mulInto(a, b, alloc(len(a)));

export const mulInto = (
  a: ComplexArray,
  b: ComplexArray,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    const ar = a.real[i]!;
    const ai = a.imag[i]!;
    const br = b.real[i]!;
    const bi = b.imag[i]!;
    out.real[i] = ar * br - ai * bi;
    out.imag[i] = ar * bi + ai * br;
  }
  return out;
};

// ── mul by complex scalar ────────────────────────────────────────────

/**
 * Multiply every element by a single complex scalar `{ re, im }`.
 */
export const mulScalar = (
  a: ComplexArray,
  re: number,
  im: number,
): ComplexArray => mulScalarInto(a, re, im, alloc(len(a)));

export const mulScalarInto = (
  a: ComplexArray,
  re: number,
  im: number,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    const ar = a.real[i]!;
    const ai = a.imag[i]!;
    out.real[i] = ar * re - ai * im;
    out.imag[i] = ar * im + ai * re;
  }
  return out;
};

// ── div (complex-by-complex, element-wise) ───────────────────────────

/**
 * Element-wise complex division `a / b`.
 * `(a + ib) / (c + id) = ((ac + bd) + i(bc − ad)) / (c² + d²)`
 */
export const div = (a: ComplexArray, b: ComplexArray): ComplexArray =>
  divInto(a, b, alloc(len(a)));

export const divInto = (
  a: ComplexArray,
  b: ComplexArray,
  out: ComplexArray,
): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    const ar = a.real[i]!;
    const ai = a.imag[i]!;
    const br = b.real[i]!;
    const bi = b.imag[i]!;
    const denom = br * br + bi * bi;
    out.real[i] = (ar * br + ai * bi) / denom;
    out.imag[i] = (ai * br - ar * bi) / denom;
  }
  return out;
};

// ── div by complex scalar ────────────────────────────────────────────

/**
 * Divide every element by a single complex scalar `{ re, im }`.
 */
export const divScalar = (
  a: ComplexArray,
  re: number,
  im: number,
): ComplexArray => divScalarInto(a, re, im, alloc(len(a)));

export const divScalarInto = (
  a: ComplexArray,
  re: number,
  im: number,
  out: ComplexArray,
): ComplexArray => {
  const denom = re * re + im * im;
  const invRe = re / denom;
  const invIm = -im / denom;
  return mulScalarInto(a, invRe, invIm, out);
};

// ── conj (complex conjugate) ─────────────────────────────────────────

/** Element-wise complex conjugate: negate imaginary parts. */
export const conj = (a: ComplexArray): ComplexArray =>
  conjInto(a, alloc(len(a)));

export const conjInto = (a: ComplexArray, out: ComplexArray): ComplexArray => {
  const n = len(a);
  for (let i = 0; i < n; i += 1) {
    out.real[i] = a.real[i]!;
    out.imag[i] = -(a.imag[i]!);
  }
  return out;
};

// ── magnitude / phase (projections — non-invertible) ─────────────────

/** Element-wise magnitude. Returns a real-valued Float64Array. */
export const mag = (a: ComplexArray, out?: Float64Array): Float64Array => {
  const n = len(a);
  const result = out ?? new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    result[i] = Math.hypot(a.real[i]!, a.imag[i]!);
  }
  return result;
};

/** Element-wise phase (argument). Returns a real-valued Float64Array. */
export const arg = (a: ComplexArray, out?: Float64Array): Float64Array => {
  const n = len(a);
  const result = out ?? new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    result[i] = Math.atan2(a.imag[i]!, a.real[i]!);
  }
  return result;
};

// ── copy / zero ──────────────────────────────────────────────────────

/** Deep copy a ComplexArray. */
export const copy = (a: ComplexArray): ComplexArray => ({
  real: new Float64Array(a.real),
  imag: new Float64Array(a.imag),
});

/** Copy `a` into `out`. */
export const copyInto = (a: ComplexArray, out: ComplexArray): ComplexArray => {
  out.real.set(a.real);
  out.imag.set(a.imag);
  return out;
};

/** Fill a ComplexArray with zeros. */
export const zero = (a: ComplexArray): ComplexArray => {
  a.real.fill(0);
  a.imag.fill(0);
  return a;
};
