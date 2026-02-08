import { describe, expect, it } from "vitest";
import type { ComplexArray } from "../../src/core/fft.js";
import {
  scale,
  scaleInto,
  add,
  addInto,
  sub,
  subInto,
  mul,
  mulInto,
  mulScalar,
  mulScalarInto,
  div,
  divInto,
  divScalar,
  divScalarInto,
  conj,
  conjInto,
  mag,
  arg,
  copy,
  copyInto,
  zero,
} from "../../src/math/complex.js";

// ── helpers ──────────────────────────────────────────────────────────

const cx = (re: number[], im: number[]): ComplexArray => ({
  real: Float64Array.from(re),
  imag: Float64Array.from(im),
});

const expectClose = (
  actual: ComplexArray | Float64Array,
  expected: number[],
  tol = 1e-12,
  field?: "real" | "imag",
) => {
  const arr =
    actual instanceof Float64Array
      ? actual
      : field === "imag"
        ? actual.imag
        : actual.real;
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(arr[i]).toBeCloseTo(expected[i]!, tol > 1e-6 ? 6 : 12);
  }
};

// ── scale ────────────────────────────────────────────────────────────

describe("scale", () => {
  it("multiplies all elements by a real scalar", () => {
    const a = cx([1, 2, 3], [4, 5, 6]);
    const r = scale(a, 2);
    expectClose(r, [2, 4, 6], 1e-12, "real");
    expectClose(r, [8, 10, 12], 1e-12, "imag");
  });

  it("does not mutate input", () => {
    const a = cx([1], [2]);
    scale(a, 10);
    expect(a.real[0]).toBe(1);
    expect(a.imag[0]).toBe(2);
  });

  it("scaleInto writes into out", () => {
    const a = cx([1, 2], [3, 4]);
    const out = cx([0, 0], [0, 0]);
    const r = scaleInto(a, 3, out);
    expect(r).toBe(out);
    expectClose(out, [3, 6], 1e-12, "real");
    expectClose(out, [9, 12], 1e-12, "imag");
  });
});

// ── add ──────────────────────────────────────────────────────────────

describe("add", () => {
  it("adds element-wise", () => {
    const a = cx([1, 2], [3, 4]);
    const b = cx([10, 20], [30, 40]);
    const r = add(a, b);
    expectClose(r, [11, 22], 1e-12, "real");
    expectClose(r, [33, 44], 1e-12, "imag");
  });

  it("addInto writes into out", () => {
    const a = cx([1], [2]);
    const b = cx([3], [4]);
    const out = cx([0], [0]);
    const r = addInto(a, b, out);
    expect(r).toBe(out);
    expect(out.real[0]).toBeCloseTo(4);
    expect(out.imag[0]).toBeCloseTo(6);
  });
});

// ── sub ──────────────────────────────────────────────────────────────

describe("sub", () => {
  it("subtracts element-wise", () => {
    const a = cx([10, 20], [30, 40]);
    const b = cx([1, 2], [3, 4]);
    const r = sub(a, b);
    expectClose(r, [9, 18], 1e-12, "real");
    expectClose(r, [27, 36], 1e-12, "imag");
  });
});

// ── mul (complex-by-complex) ─────────────────────────────────────────

describe("mul", () => {
  it("computes Hadamard product", () => {
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    const a = cx([1], [2]);
    const b = cx([3], [4]);
    const r = mul(a, b);
    expect(r.real[0]).toBeCloseTo(-5);
    expect(r.imag[0]).toBeCloseTo(10);
  });

  it("mulInto writes into out", () => {
    const a = cx([1], [2]);
    const b = cx([3], [4]);
    const out = cx([0], [0]);
    const r = mulInto(a, b, out);
    expect(r).toBe(out);
    expect(out.real[0]).toBeCloseTo(-5);
    expect(out.imag[0]).toBeCloseTo(10);
  });
});

// ── mulScalar ────────────────────────────────────────────────────────

describe("mulScalar", () => {
  it("multiplies every element by a complex scalar", () => {
    // (2+3i) * (0+1i) = -3 + 2i
    const a = cx([2], [3]);
    const r = mulScalar(a, 0, 1);
    expect(r.real[0]).toBeCloseTo(-3);
    expect(r.imag[0]).toBeCloseTo(2);
  });
});

// ── div ──────────────────────────────────────────────────────────────

describe("div", () => {
  it("divides element-wise", () => {
    // (-5+10i) / (3+4i) = ((−5)(3)+(10)(4) + i((10)(3)−(−5)(4))) / (9+16)
    //                    = (−15+40 + i(30+20)) / 25
    //                    = 25/25 + 50i/25 = 1 + 2i
    const a = cx([-5], [10]);
    const b = cx([3], [4]);
    const r = div(a, b);
    expect(r.real[0]).toBeCloseTo(1);
    expect(r.imag[0]).toBeCloseTo(2);
  });

  it("divInto writes into out", () => {
    const a = cx([-5], [10]);
    const b = cx([3], [4]);
    const out = cx([0], [0]);
    divInto(a, b, out);
    expect(out.real[0]).toBeCloseTo(1);
    expect(out.imag[0]).toBeCloseTo(2);
  });
});

// ── divScalar ────────────────────────────────────────────────────────

describe("divScalar", () => {
  it("divides every element by a complex scalar", () => {
    // (1+2i) / (0+1i) = (1+2i)(-i) = (1)(-i) + (2i)(-i) = -i + 2 = 2 - i
    const a = cx([1], [2]);
    const r = divScalar(a, 0, 1);
    expect(r.real[0]).toBeCloseTo(2);
    expect(r.imag[0]).toBeCloseTo(-1);
  });
});

// ── conj ─────────────────────────────────────────────────────────────

describe("conj", () => {
  it("negates imaginary parts", () => {
    const a = cx([1, 2], [3, -4]);
    const r = conj(a);
    expectClose(r, [1, 2], 1e-12, "real");
    expectClose(r, [-3, 4], 1e-12, "imag");
  });

  it("is its own inverse", () => {
    const a = cx([1, 2], [3, -4]);
    const r = conj(conj(a));
    expectClose(r, [1, 2], 1e-12, "real");
    expectClose(r, [3, -4], 1e-12, "imag");
  });

  it("conjInto can write in-place", () => {
    const a = cx([5], [7]);
    conjInto(a, a);
    expect(a.real[0]).toBe(5);
    expect(a.imag[0]).toBe(-7);
  });
});

// ── mag / arg ────────────────────────────────────────────────────────

describe("mag", () => {
  it("computes magnitude", () => {
    const a = cx([3], [4]);
    const r = mag(a);
    expect(r[0]).toBeCloseTo(5);
  });
});

describe("arg", () => {
  it("computes phase", () => {
    const a = cx([1], [1]);
    const r = arg(a);
    expect(r[0]).toBeCloseTo(Math.PI / 4);
  });
});

// ── copy / zero ──────────────────────────────────────────────────────

describe("copy", () => {
  it("deep copies", () => {
    const a = cx([1, 2], [3, 4]);
    const r = copy(a);
    expect(r.real).not.toBe(a.real);
    expect(r.imag).not.toBe(a.imag);
    expectClose(r, [1, 2], 1e-12, "real");
    expectClose(r, [3, 4], 1e-12, "imag");
  });

  it("copyInto writes into target", () => {
    const a = cx([1], [2]);
    const out = cx([0], [0]);
    copyInto(a, out);
    expect(out.real[0]).toBe(1);
    expect(out.imag[0]).toBe(2);
  });
});

describe("zero", () => {
  it("fills with zeros in-place", () => {
    const a = cx([1, 2], [3, 4]);
    const r = zero(a);
    expect(r).toBe(a);
    expect(a.real[0]).toBe(0);
    expect(a.imag[0]).toBe(0);
  });
});
