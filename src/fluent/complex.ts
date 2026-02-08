/**
 * Fluent, mutating-by-default wrapper over ComplexArray with typestate generics.
 *
 * The typestate parameter `S` tracks:
 *   - kind:   "complex" | "real"  — whether full complex info is preserved
 *   - hasFft: true | false        — whether an inverse transform context is available
 *   - invert: "yes" | "no" | "maybe" — whether the chain is provably invertible
 *   - len:    "same" | "changed"  — whether the original length is preserved
 *
 * `.inverse()` is gated via a `this:` constraint: it only exists when the
 * typestate satisfies `InverseReady`.
 *
 * **All chainable operations mutate the underlying buffers** and return `this`.
 * Use `.clone()` to get an independent copy before mutating if you need persistence.
 *
 * @module
 */

import type { ComplexArray } from "../core/fft.js";
import {
  scaleInto,
  mulInto,
  mulScalarInto,
  divInto,
  divScalarInto,
  conjInto,
  addInto,
  subInto,
  copy,
  mag as magFn,
  arg as argFn,
} from "../math/complex.js";

// ── Typestate types ──────────────────────────────────────────────────

/** State dimensions tracked by the fluent chain. */
export type ChainState = {
  kind: "complex" | "real";
  hasFft: boolean;
  invert: "yes" | "no" | "maybe";
  len: "same" | "changed";
};

/**
 * The "default" state for a chain created from raw complex data
 * (no FFT context).
 */
export type DefaultState = {
  kind: "complex";
  hasFft: false;
  invert: "yes";
  len: "same";
};

/**
 * State produced by `FluentFFT.forward()` — has inverse context.
 */
export type FftForwardState = {
  kind: "complex";
  hasFft: true;
  invert: "yes";
  len: "same";
};

/**
 * The state required for `.inverse()` to be callable.
 */
export type InverseReady = {
  kind: "complex";
  hasFft: true;
  invert: "yes";
  len: "same";
};

// ── Branded NonZero ──────────────────────────────────────────────────

declare const _nonzero: unique symbol;

/**
 * Branded number type that is statically known to be nonzero.
 * Create via `assertNonZero()` or `asNonZero()`.
 */
export type NonZero = number & { readonly [_nonzero]: true };

/**
 * Assertion helper: throws if `x === 0`, otherwise narrows to `NonZero`.
 */
export function assertNonZero(x: number): asserts x is NonZero {
  if (x === 0) throw new Error("Expected nonzero value, got 0");
}

/**
 * Safe constructor: returns `NonZero` or `null`.
 */
export const asNonZero = (x: number): NonZero | null =>
  x !== 0 ? (x as NonZero) : null;

// ── Inverse error ────────────────────────────────────────────────────

export type InverseError =
  | { readonly _tag: "NoFftContext" }
  | { readonly _tag: "NotInvertible"; readonly reason: string }
  | { readonly _tag: "LengthMismatch"; readonly expected: number; readonly actual: number };

export type InverseResult =
  | { readonly ok: true; readonly value: ComplexArray }
  | { readonly ok: false; readonly error: InverseError };

// ── InverseFn type (injected by FluentFFT) ───────────────────────────

export type InverseFn = (input: ComplexArray, out?: ComplexArray) => ComplexArray;

// ── ComplexChain class ───────────────────────────────────────────────

/**
 * Fluent, mutating wrapper over a `ComplexArray`.
 *
 * **Mutations**: all chainable ops mutate the underlying `data` in-place and
 * return `this`. Call `.clone()` first if you need an independent copy.
 *
 * @typeParam S  Typestate tracking kind, FFT context, invertibility, and length.
 */
export class ComplexChain<S extends ChainState> {
  /** The underlying complex data. Mutated in-place by chainable ops. */
  readonly data: ComplexArray;

  /**
   * Optional inverse transform callback, injected by `FluentFFT.forward()`.
   * @internal
   */
  readonly _inverseFn: InverseFn | null;

  constructor(data: ComplexArray, inverseFn: InverseFn | null = null) {
    this.data = data;
    this._inverseFn = inverseFn;
  }

  // ── identity / accessors ─────────────────────────────────────────

  /** Return the underlying `{ real, imag }` without copying. */
  unwrap(): ComplexArray {
    return this.data;
  }

  /** Number of complex elements. */
  get length(): number {
    return this.data.real.length;
  }

  // ── persistence helpers ──────────────────────────────────────────

  /** Deep copy of the chain (preserves typestate and inverse context). */
  clone(): ComplexChain<S> {
    return new ComplexChain<S>(copy(this.data), this._inverseFn);
  }

  // ── invertible ops (state stays "yes" or keeps current) ──────────

  /**
   * Multiply every element by a real scalar.
   *
   * - If `s` is `NonZero`, invertibility is preserved.
   * - If `s` is a plain `number`, invertibility becomes `"maybe"`.
   */
  scale(
    s: NonZero,
  ): ComplexChain<S>;
  scale(
    s: number,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }>;
  scale(s: number): ComplexChain<any> {
    scaleInto(this.data, s, this.data);
    return this;
  }

  /**
   * Element-wise complex multiply (Hadamard) with another complex vector.
   * Invertibility becomes "maybe" (runtime-dependent on whether `b` has zeros).
   */
  mul(
    b: ComplexArray,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }> {
    mulInto(this.data, b, this.data);
    return this as any;
  }

  /**
   * Multiply every element by a complex scalar `(re, im)`.
   */
  mulScalar(
    re: NonZero,
    im: number,
  ): ComplexChain<S>;
  mulScalar(
    re: number,
    im: NonZero,
  ): ComplexChain<S>;
  mulScalar(
    re: number,
    im: number,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }>;
  mulScalar(re: number, im: number): ComplexChain<any> {
    mulScalarInto(this.data, re, im, this.data);
    return this;
  }

  /**
   * Element-wise complex division.
   * Invertibility becomes "maybe" (runtime-dependent on zeros).
   */
  div(
    b: ComplexArray,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }> {
    divInto(this.data, b, this.data);
    return this as any;
  }

  /**
   * Divide every element by a complex scalar `(re, im)`.
   */
  divScalar(
    re: NonZero,
    im: number,
  ): ComplexChain<S>;
  divScalar(
    re: number,
    im: NonZero,
  ): ComplexChain<S>;
  divScalar(
    re: number,
    im: number,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }>;
  divScalar(re: number, im: number): ComplexChain<any> {
    divScalarInto(this.data, re, im, this.data);
    return this;
  }

  /** Complex conjugate (self-inverse). Preserves invertibility. */
  conj(): ComplexChain<S> {
    conjInto(this.data, this.data);
    return this;
  }

  /** Element-wise add. Invertibility becomes "maybe". */
  add(
    b: ComplexArray,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }> {
    addInto(this.data, b, this.data);
    return this as any;
  }

  /** Element-wise subtract. Invertibility becomes "maybe". */
  sub(
    b: ComplexArray,
  ): ComplexChain<Omit<S, "invert"> & { invert: "maybe" }> {
    subInto(this.data, b, this.data);
    return this as any;
  }

  // ── non-invertible projections ───────────────────────────────────

  /**
   * Compute element-wise magnitude. Returns a plain Float64Array.
   * This is a terminal projection — it destroys phase info and
   * returns raw data (no further chaining on complex).
   */
  mag(): Float64Array {
    return magFn(this.data);
  }

  /**
   * Compute element-wise phase (argument). Returns a plain Float64Array.
   * This is a terminal projection.
   */
  arg(): Float64Array {
    return argFn(this.data);
  }

  // ── inverse (typestate-gated) ────────────────────────────────────

  /**
   * Apply the inverse transform (IFFT) and return the result.
   *
   * **Only callable when the chain's typestate is `InverseReady`:**
   * - `kind === "complex"` (no info-losing projections applied)
   * - `hasFft === true` (chain was created from `FluentFFT.forward()`)
   * - `invert === "yes"` (no known-destructive ops applied)
   * - `len === "same"` (no shape changes)
   *
   * If TypeScript rejects this call, it means the chain has lost the
   * information needed for a faithful round-trip inverse.
   */
  inverse(
    this: ComplexChain<InverseReady>,
    out?: ComplexArray,
  ): ComplexArray {
    return this._inverseFn!(this.data, out);
  }

  /**
   * Checked inverse: always callable when `hasFft` is true, regardless of
   * `invert` state. Returns a result union so runtime errors are explicit.
   */
  inverseChecked(
    this: ComplexChain<S & { hasFft: true }>,
    out?: ComplexArray,
  ): InverseResult {
    try {
      const value = this._inverseFn!(this.data, out);
      return { ok: true, value };
    } catch (e) {
      return {
        ok: false,
        error: {
          _tag: "NotInvertible",
          reason: e instanceof Error ? e.message : String(e),
        },
      };
    }
  }
}

// ── Factory for wrapping raw data (no FFT context) ───────────────────

/**
 * Wrap an existing `ComplexArray` in a fluent chain without FFT context.
 * The chain starts as `{ kind:"complex", hasFft:false, invert:"yes", len:"same" }`.
 *
 * Use `FluentFFT.forward()` if you need `.inverse()`.
 */
export const chain = (data: ComplexArray): ComplexChain<DefaultState> =>
  new ComplexChain<DefaultState>(data);
