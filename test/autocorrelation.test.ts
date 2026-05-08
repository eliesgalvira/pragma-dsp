import { describe, expect, it } from "vitest";
import { autocorrelation } from "../src/xform/fourier.ts";

describe("autocorrelation()", () => {
  it("matches basic linear autocorrelation for small lags", () => {
    const samples = Float64Array.from([1, 2, 3]);
    const r = autocorrelation(samples);

    // r[0] = sum x[i]^2
    expect(r[0]).toBeCloseTo(14, 8);
    // r[1] = x0*x1 + x1*x2
    expect(r[1]).toBeCloseTo(8, 8);
    // r[2] = x0*x2
    expect(r[2]).toBeCloseTo(3, 8);
  });
});
