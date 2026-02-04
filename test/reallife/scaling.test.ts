import { describe, expect, it } from "vitest";
import { spectrum } from "../../src/public/spectrum.js";
import { loadPureSine, loadSpecial } from "./helpers.js";

describe("spectrum() amplitude scaling", () => {
  describe("one-sided spectrum (default)", () => {
    const ref = loadPureSine();

    // Filter to bin-centered cases for precise amplitude testing
    const binCenteredCases = ref.cases.filter(
      (c) => c.kind === "pure_sine_bin_centered"
    );

    for (const testCase of binCenteredCases) {
      it(`returns correct amplitude for ${testCase.name}`, () => {
        const result = spectrum(testCase.signal, {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: "rect",
          sides: "one"
        });

        const expectedAmp = testCase.params.amplitude as number;
        const expectedBin = testCase.params.bin_index as number;

        // Peak should be at the expected bin
        expect(result.peak.index).toBe(expectedBin);

        // Amplitude should match input amplitude (within tolerance)
        // The scaling formula doubles non-DC/non-Nyquist bins
        expect(result.peak.amplitude).toBeCloseTo(expectedAmp, 2);
      });
    }

    it("DC bin is not doubled", () => {
      const special = loadSpecial();
      const dcCase = special.cases.find((c) => c.kind === "dc");
      if (!dcCase) throw new Error("Missing DC test case");

      const result = spectrum(dcCase.signal, {
        sampleRate: dcCase.sampleRate,
        fftSize: dcCase.n,
        window: "rect",
        sides: "one"
      });

      // DC amplitude should be level (not 2*level)
      expect(result.amplitude[0]).toBeCloseTo(dcCase.params.level as number, 6);
    });

    it("Nyquist bin is not doubled", () => {
      const special = loadSpecial();
      const nyquistCase = special.cases.find((c) => c.kind === "nyquist");
      if (!nyquistCase) throw new Error("Missing Nyquist test case");

      const result = spectrum(nyquistCase.signal, {
        sampleRate: nyquistCase.sampleRate,
        fftSize: nyquistCase.n,
        window: "rect",
        sides: "one"
      });

      // Nyquist bin is at index N/2 in the one-sided spectrum
      const nyquistIndex = nyquistCase.n / 2;
      // Nyquist amplitude should be amplitude (not 2*amplitude)
      expect(result.amplitude[nyquistIndex]).toBeCloseTo(
        nyquistCase.params.amplitude as number,
        6
      );
    });
  });

  describe("two-sided spectrum", () => {
    const ref = loadPureSine();

    const binCenteredCases = ref.cases.filter(
      (c) => c.kind === "pure_sine_bin_centered"
    );

    for (const testCase of binCenteredCases) {
      it(`returns correct amplitude for ${testCase.name} (two-sided)`, () => {
        const result = spectrum(testCase.signal, {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: "rect",
          sides: "two"
        });

        const expectedAmp = testCase.params.amplitude as number;
        const expectedBin = testCase.params.bin_index as number;

        // In two-sided, energy is split between positive and negative frequencies
        // Each side should have amplitude/2
        expect(result.amplitude[expectedBin]).toBeCloseTo(expectedAmp / 2, 2);

        // Negative frequency bin
        const negativeBin = testCase.n - expectedBin;
        expect(result.amplitude[negativeBin]).toBeCloseTo(expectedAmp / 2, 2);
      });
    }

    it("returns full N bins for two-sided", () => {
      const testCase = ref.cases[0];
      if (!testCase) throw new Error("No test cases available");

      const result = spectrum(testCase.signal, {
        sampleRate: testCase.sampleRate,
        fftSize: testCase.n,
        window: "rect",
        sides: "two"
      });

      expect(result.amplitude.length).toBe(testCase.n);
      expect(result.frequencies.length).toBe(testCase.n);
      expect(result.phase.length).toBe(testCase.n);
    });
  });

  describe("frequency axis", () => {
    const ref = loadPureSine();

    for (const testCase of ref.cases.filter(
      (c) => c.kind === "pure_sine_bin_centered"
    )) {
      it(`peak frequency matches expected for ${testCase.name}`, () => {
        const result = spectrum(testCase.signal, {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: "rect",
          sides: "one"
        });

        const expectedFreq = testCase.params.frequency_hz as number;
        expect(result.peak.frequency).toBeCloseTo(expectedFreq, 6);
      });
    }

    it("frequency axis is correctly scaled", () => {
      const testCase = ref.cases[0];
      if (!testCase) throw new Error("No test cases available");

      const result = spectrum(testCase.signal, {
        sampleRate: testCase.sampleRate,
        fftSize: testCase.n,
        window: "rect",
        sides: "one"
      });

      // First bin is DC (0 Hz)
      expect(result.frequencies[0]).toBe(0);

      // Frequency spacing is sampleRate / fftSize
      const binWidth = testCase.sampleRate / testCase.n;
      for (let i = 0; i < result.frequencies.length; i += 1) {
        expect(result.frequencies[i]).toBeCloseTo(i * binWidth, 10);
      }

      // Last bin is Nyquist (sampleRate / 2)
      const nyquist = testCase.sampleRate / 2;
      expect(result.frequencies[result.frequencies.length - 1]).toBeCloseTo(
        nyquist,
        10
      );
    });
  });

  describe("peak detection", () => {
    it("ignores DC when there are non-DC components", () => {
      const special = loadSpecial();
      const dcPlusSine = special.cases.find((c) => c.kind === "dc_plus_sine");
      if (!dcPlusSine) throw new Error("Missing dc_plus_sine test case");

      const result = spectrum(dcPlusSine.signal, {
        sampleRate: dcPlusSine.sampleRate,
        fftSize: dcPlusSine.n,
        window: "rect",
        sides: "one"
      });

      // Peak should be at the sine bin, not the DC bin
      const expectedBin = dcPlusSine.params.sine_bin as number;
      expect(result.peak.index).toBe(expectedBin);
    });

    it("returns DC as peak when it is the only component", () => {
      const special = loadSpecial();
      const dcCase = special.cases.find((c) => c.kind === "dc");
      if (!dcCase) throw new Error("Missing DC test case");

      const result = spectrum(dcCase.signal, {
        sampleRate: dcCase.sampleRate,
        fftSize: dcCase.n,
        window: "rect",
        sides: "one"
      });

      // For pure DC, the "peak" finder should return DC since there's nothing else
      // Note: the current implementation prefers non-DC if any exists
      // For pure DC, nonDcValue will be 0, so maxIndex (0) is returned
      expect(result.peak.index).toBe(0);
    });
  });
});
