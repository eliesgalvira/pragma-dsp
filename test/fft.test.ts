import { describe, expect, it } from "vitest";
import { FFT } from "../src/xform/fourier.js";
import { getCasesByN, loadFixtures } from "./fixtures.js";

const expectCloseArray = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tol = 1e-6
) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    const a = actual[i] ?? 0;
    const e = expected[i] ?? 0;
    expect(Math.abs(a - e)).toBeLessThanOrEqual(tol);
  }
};

describe("FFT fixtures", () => {
  const fixtures = loadFixtures();
  const sizes = [8, 16, 32];

  for (const n of sizes) {
    const cases = getCasesByN(fixtures, n).filter(
      (c) => c.kind === "random_normal"
    );
    for (const fixture of cases) {
      it(`matches numpy fft for ${fixture.name}`, () => {
        const fft = new FFT(fixture.n);
        const result = fft.forward(fixture.input);
        expectCloseArray(result.real, fixture.fftRe);
        expectCloseArray(result.imag, fixture.fftIm);
      });

      it(`round-trips ${fixture.name}`, () => {
        const fft = new FFT(fixture.n);
        const result = fft.forward(fixture.input);
        const roundTrip = fft.inverse(result);
        expectCloseArray(roundTrip.real, fixture.input, 1e-6);
        expectCloseArray(roundTrip.imag, new Array(fixture.n).fill(0), 1e-6);
      });
    }
  }
});
