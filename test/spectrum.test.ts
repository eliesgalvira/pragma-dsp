import { describe, expect, it } from "vitest";
import { spectrum } from "../src/public/spectrum.js";
import { loadFixtures } from "./fixtures.js";

describe("spectrum scaling + peak detection", () => {
  const fixtures = loadFixtures();
  const sineCase = fixtures.fftCases.find(
    (c) => c.kind === "sine_bin_centered"
  );

  if (!sineCase) {
    throw new Error("Missing sine_bin_centered fixture case.");
  }

  it("returns correct peak bin + frequency + amplitude", () => {
    const result = spectrum(sineCase.input, {
      sampleRate: sineCase.sampleRate,
      fftSize: sineCase.n,
      window: "rect",
      sides: "one"
    });

    const expectedK = sineCase.meta.binCenteredK ?? 0;
    const expectedHz = sineCase.meta.expectedPeakHz ?? 0;
    const expectedAmp = sineCase.meta.amplitude ?? 0;

    expect(result.peak.index).toBe(expectedK);
    expect(Math.abs(result.peak.frequency - expectedHz)).toBeLessThanOrEqual(
      1e-6
    );
    expect(Math.abs(result.peak.amplitude - expectedAmp)).toBeLessThanOrEqual(
      1e-3
    );
  });

  it("returns raw complex spectrum when requested", () => {
    const input = Float64Array.from([0, 1, 0, -1, 0, 1, 0, -1]);
    const result = spectrum(input, { fftSize: 8, returnComplex: true });

    expect(result.complex.real.length).toBe(8);
    expect(result.complex.imag.length).toBe(8);
  });
});
