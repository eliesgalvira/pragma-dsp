import { describe, expect, it } from "vitest";
import { FluentFFT } from "../../src/xform/fourier-fluent.js";
import {
  chain,
  assertNonZero,
  asNonZero,
  type NonZero,
  type InverseReady,
  ComplexChain,
} from "../../src/fluent/complex.js";
import { createComplexArray } from "../../src/core/fft.js";

// ── helpers ──────────────────────────────────────────────────────────

const expectCloseArr = (actual: Float64Array, expected: number[], tol = 1e-10) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i]!, 8);
  }
};

// ── FluentFFT basic round-trip ───────────────────────────────────────

describe("FluentFFT", () => {
  it("forward returns a ComplexChain with inverse context", () => {
    const fft = new FluentFFT(8);
    const signal = Float64Array.from([0, 1, 0, -1, 0, 1, 0, -1]);
    const c = fft.forward(signal);
    expect(c).toBeInstanceOf(ComplexChain);
    expect(c._inverseFn).not.toBeNull();
  });

  it("round-trips a simple signal via fluent chain", () => {
    const fft = new FluentFFT(8);
    const signal = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = fft.forward(signal).inverse();
    expectCloseArr(result.real, [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("round-trips after conj().conj() (self-inverse)", () => {
    const fft = new FluentFFT(8);
    const signal = Float64Array.from([1, 0, -1, 0, 1, 0, -1, 0]);
    const result = fft.forward(signal).conj().conj().inverse();
    expectCloseArr(result.real, [1, 0, -1, 0, 1, 0, -1, 0]);
  });

  it("scale + inverse round-trips with NonZero scalar", () => {
    const fft = new FluentFFT(4);
    const signal = Float64Array.from([1, 2, 3, 4]);
    const s = 3 as NonZero;

    // Scale by 3 in freq domain, then scale back by 1/3, then inverse
    const invS = (1 / 3) as NonZero;
    const result = fft.forward(signal).scale(s).scale(invS).inverse();
    expectCloseArr(result.real, [1, 2, 3, 4]);
  });

  it("forwardComplex works", () => {
    const fft = new FluentFFT(4);
    const input = createComplexArray(4);
    input.real.set([1, 0, -1, 0]);
    const c = fft.forwardComplex(input);
    expect(c).toBeInstanceOf(ComplexChain);
    const result = c.inverse();
    expectCloseArr(result.real, [1, 0, -1, 0]);
  });

  it("inverse with out parameter writes into provided buffer", () => {
    const fft = new FluentFFT(4);
    const signal = Float64Array.from([5, 6, 7, 8]);
    const out = createComplexArray(4);
    const result = fft.forward(signal).inverse(out);
    expect(result).toBe(out);
    expectCloseArr(out.real, [5, 6, 7, 8]);
  });
});

// ── chain() factory (no FFT context) ─────────────────────────────────

describe("chain()", () => {
  it("creates a chain from raw data", () => {
    const data = createComplexArray(4);
    data.real.set([1, 2, 3, 4]);
    const c = chain(data);
    expect(c.unwrap()).toBe(data);
    expect(c.length).toBe(4);
  });

  it("mutates in-place by default", () => {
    const data = createComplexArray(2);
    data.real.set([1, 2]);
    data.imag.set([3, 4]);

    chain(data).scale(2 as NonZero);

    // Underlying data is mutated
    expect(data.real[0]).toBeCloseTo(2);
    expect(data.real[1]).toBeCloseTo(4);
    expect(data.imag[0]).toBeCloseTo(6);
    expect(data.imag[1]).toBeCloseTo(8);
  });

  it("clone() creates an independent copy", () => {
    const data = createComplexArray(2);
    data.real.set([1, 2]);
    data.imag.set([3, 4]);

    const c = chain(data);
    const cloned = c.clone();

    cloned.scale(10 as NonZero);
    // Original is not mutated
    expect(data.real[0]).toBe(1);
    expect(data.imag[0]).toBe(3);
  });
});

// ── fluent chaining ops ──────────────────────────────────────────────

describe("fluent ops", () => {
  it("chains scale + conj + scale correctly", () => {
    const data = createComplexArray(2);
    data.real.set([1, 2]);
    data.imag.set([3, 4]);

    const s = 2 as NonZero;
    // (1+3i)*2 = 2+6i → conj → 2−6i → *2 = 4−12i
    // (2+4i)*2 = 4+8i → conj → 4−8i → *2 = 8−16i
    chain(data).scale(s).conj().scale(s);

    expect(data.real[0]).toBeCloseTo(4);
    expect(data.real[1]).toBeCloseTo(8);
    expect(data.imag[0]).toBeCloseTo(-12);
    expect(data.imag[1]).toBeCloseTo(-16);
  });

  it("mul() does element-wise complex multiply", () => {
    const data = createComplexArray(1);
    data.real.set([1]);
    data.imag.set([2]);

    const other = createComplexArray(1);
    other.real.set([3]);
    other.imag.set([4]);

    // (1+2i)(3+4i) = -5 + 10i
    chain(data).mul(other);
    expect(data.real[0]).toBeCloseTo(-5);
    expect(data.imag[0]).toBeCloseTo(10);
  });

  it("div() does element-wise complex divide", () => {
    const data = createComplexArray(1);
    data.real.set([-5]);
    data.imag.set([10]);

    const other = createComplexArray(1);
    other.real.set([3]);
    other.imag.set([4]);

    // (-5+10i)/(3+4i) = 1+2i
    chain(data).div(other);
    expect(data.real[0]).toBeCloseTo(1);
    expect(data.imag[0]).toBeCloseTo(2);
  });

  it("mulScalar() multiplies by complex scalar", () => {
    const data = createComplexArray(1);
    data.real.set([2]);
    data.imag.set([3]);

    // (2+3i)(0+1i) = -3+2i
    const re = 0;
    const im = 1 as NonZero;
    chain(data).mulScalar(re, im);
    expect(data.real[0]).toBeCloseTo(-3);
    expect(data.imag[0]).toBeCloseTo(2);
  });

  it("divScalar() divides by complex scalar", () => {
    const data = createComplexArray(1);
    data.real.set([1]);
    data.imag.set([2]);

    // (1+2i)/(0+1i) = 2-1i
    const re = 0;
    const im = 1 as NonZero;
    chain(data).divScalar(re, im);
    expect(data.real[0]).toBeCloseTo(2);
    expect(data.imag[0]).toBeCloseTo(-1);
  });

  it("add/sub chain", () => {
    const data = createComplexArray(2);
    data.real.set([1, 2]);
    data.imag.set([3, 4]);

    const b = createComplexArray(2);
    b.real.set([10, 20]);
    b.imag.set([30, 40]);

    chain(data).add(b).sub(b);
    expect(data.real[0]).toBeCloseTo(1);
    expect(data.imag[0]).toBeCloseTo(3);
  });

  it("mag() returns magnitudes as terminal projection", () => {
    const data = createComplexArray(1);
    data.real.set([3]);
    data.imag.set([4]);
    const m = chain(data).mag();
    expect(m[0]).toBeCloseTo(5);
  });

  it("arg() returns phases as terminal projection", () => {
    const data = createComplexArray(1);
    data.real.set([1]);
    data.imag.set([1]);
    const p = chain(data).arg();
    expect(p[0]).toBeCloseTo(Math.PI / 4);
  });
});

// ── NonZero branded type ─────────────────────────────────────────────

describe("NonZero", () => {
  it("assertNonZero passes for nonzero", () => {
    expect(() => assertNonZero(5)).not.toThrow();
  });

  it("assertNonZero throws for zero", () => {
    expect(() => assertNonZero(0)).toThrow("Expected nonzero value, got 0");
  });

  it("asNonZero returns NonZero for nonzero", () => {
    const v = asNonZero(42);
    expect(v).toBe(42);
  });

  it("asNonZero returns null for zero", () => {
    expect(asNonZero(0)).toBeNull();
  });
});

// ── inverseChecked ───────────────────────────────────────────────────

describe("inverseChecked", () => {
  it("returns ok:true for valid inverse", () => {
    const fft = new FluentFFT(4);
    const signal = Float64Array.from([1, 2, 3, 4]);
    const s = 2 as NonZero;
    const invS = 0.5 as NonZero;

    // Build a chain that is still InverseReady
    const c = fft.forward(signal).scale(s).scale(invS);
    const result = c.inverseChecked();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expectCloseArr(result.value.real, [1, 2, 3, 4]);
    }
  });

  it("returns ok:true even for maybe-invertible chains (runtime check)", () => {
    const fft = new FluentFFT(4);
    const signal = Float64Array.from([1, 2, 3, 4]);

    // .scale(number) → invert:"maybe", but inverseChecked is still callable
    const c = fft.forward(signal).scale(2);
    const result = c.inverseChecked();
    expect(result.ok).toBe(true);
  });
});

// ── realistic DSP pipeline ───────────────────────────────────────────

describe("realistic pipeline", () => {
  it("forward → scale → conj → inverse round-trip", () => {
    const fft = new FluentFFT(8);
    const signal = Float64Array.from([0, 1, 0, -1, 0, 1, 0, -1]);
    const s = 1 as NonZero;

    const result = fft.forward(signal).scale(s).conj().conj().inverse();
    expectCloseArr(result.real, [0, 1, 0, -1, 0, 1, 0, -1]);
  });

  it("convolution via mul in frequency domain", () => {
    const N = 8;
    const fft = new FluentFFT(N);

    // Impulse signal
    const x = new Float64Array(N);
    x[0] = 1;

    // Simple filter (another impulse shifted by 1)
    const h = new Float64Array(N);
    h[1] = 1;

    // Forward both
    const X = fft.forward(x);
    const fft2 = new FluentFFT(N);
    const H = fft2.forward(h);

    // Multiply in frequency domain = circular convolution
    X.mul(H.unwrap());

    // inverseChecked because mul makes it "maybe"
    const result = X.inverseChecked();
    expect(result.ok).toBe(true);
    if (result.ok) {
      // Convolving impulse at 0 with impulse at 1 gives impulse at 1
      expect(result.value.real[1]).toBeCloseTo(1, 8);
      // Other bins near zero
      expect(Math.abs(result.value.real[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(result.value.real[2]!)).toBeLessThan(1e-10);
    }
  });
});
