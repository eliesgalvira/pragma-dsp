import { describe, expect, it } from "vitest";
import { FFT, phase } from "../../src/xform/fourier.js";
import { spectrum } from "../../src/public/spectrum.js";
import { loadCosine, loadPureSine } from "./helpers.js";

describe("phase accuracy", () => {
  describe("sine vs cosine phase difference", () => {
    const sineRef = loadPureSine();
    const cosineRef = loadCosine();

    // Find matching sine and cosine cases at the same frequency
    const sineBin8 = sineRef.cases.find(
      (c) => c.kind === "pure_sine_bin_centered" && c.params.bin_index === 8
    );
    const cosineBin8 = cosineRef.cases.find(
      (c) => c.kind === "cosine" && c.params.bin_index === 8
    );

    if (sineBin8 && cosineBin8) {
      it("cosine leads sine by 90 degrees at peak bin", () => {
        const fft = new FFT(sineBin8.n);

        const sineResult = fft.forward(sineBin8.signal);
        const cosineResult = fft.forward(cosineBin8.signal);

        const sinePhase = phase(sineResult);
        const cosinePhase = phase(cosineResult);

        const bin = sineBin8.params.bin_index as number;

        // Phase difference should be approximately π/2 (90 degrees)
        // cos(x) = sin(x + π/2), so cosine leads sine
        let phaseDiff = cosinePhase[bin]! - sinePhase[bin]!;

        // Normalize to [-π, π]
        while (phaseDiff > Math.PI) phaseDiff -= 2 * Math.PI;
        while (phaseDiff < -Math.PI) phaseDiff += 2 * Math.PI;

        // Cosine should lead by π/2
        expect(Math.abs(phaseDiff - Math.PI / 2)).toBeLessThan(1e-6);
      });
    }
  });

  describe("known phase signals", () => {
    const ref = loadPureSine();

    const phaseCases = ref.cases.filter((c) => c.kind === "pure_sine_phase");

    for (const testCase of phaseCases) {
      it(`phase is correct for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);
        const ph = phase(result);

        const bin = testCase.params.bin_index as number;
        const inputPhaseRad = testCase.params.phase_rad as number;

        // For sin(2πft + φ), the FFT phase at positive frequency bin k should be:
        // -π/2 - φ (since sin = cos shifted by -π/2)
        // But this depends on the exact FFT convention.
        //
        // More practically, we just verify it matches NumPy:
        const expectedPhase = testCase.phase[bin]!;

        let diff = Math.abs(ph[bin]! - expectedPhase);
        // Handle phase wrapping
        diff = Math.min(diff, Math.abs(diff - 2 * Math.PI));

        expect(diff).toBeLessThan(1e-10);
      });
    }
  });

  describe("spectrum() phase output", () => {
    const ref = loadPureSine();

    const phaseCases = ref.cases.filter((c) => c.kind === "pure_sine_phase");

    for (const testCase of phaseCases) {
      it(`spectrum() reports correct peak phase for ${testCase.name}`, () => {
        const result = spectrum(testCase.signal, {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: "rect",
          sides: "one"
        });

        const expectedBin = testCase.params.bin_index as number;
        const expectedPhase = testCase.phase[expectedBin]!;

        expect(result.peak.index).toBe(expectedBin);

        let diff = Math.abs(result.peak.phase - expectedPhase);
        diff = Math.min(diff, Math.abs(diff - 2 * Math.PI));

        expect(diff).toBeLessThan(1e-10);
      });
    }
  });

  describe("phase continuity", () => {
    it("phase array has correct length for one-sided spectrum", () => {
      const ref = loadPureSine();
      const testCase = ref.cases[0];
      if (!testCase) throw new Error("No test cases");

      const result = spectrum(testCase.signal, {
        sampleRate: testCase.sampleRate,
        fftSize: testCase.n,
        window: "rect",
        sides: "one"
      });

      // One-sided: N/2 + 1 bins
      expect(result.phase.length).toBe(testCase.n / 2 + 1);
    });

    it("phase array has correct length for two-sided spectrum", () => {
      const ref = loadPureSine();
      const testCase = ref.cases[0];
      if (!testCase) throw new Error("No test cases");

      const result = spectrum(testCase.signal, {
        sampleRate: testCase.sampleRate,
        fftSize: testCase.n,
        window: "rect",
        sides: "two"
      });

      // Two-sided: N bins
      expect(result.phase.length).toBe(testCase.n);
    });
  });

  describe("zero-phase at DC for symmetric signal", () => {
    it("DC phase is 0 for positive DC signal", () => {
      // A positive constant signal has real FFT output at DC, so phase = 0
      const n = 64;
      const signal = new Float64Array(n).fill(1.0);

      const fft = new FFT(n);
      const result = fft.forward(signal);
      const ph = phase(result);

      expect(ph[0]).toBeCloseTo(0, 10);
    });

    it("DC phase is π for negative DC signal", () => {
      // A negative constant signal has negative real FFT output at DC, so phase = ±π
      const n = 64;
      const signal = new Float64Array(n).fill(-1.0);

      const fft = new FFT(n);
      const result = fft.forward(signal);
      const ph = phase(result);

      // atan2(0, negative) = π
      expect(Math.abs(ph[0])).toBeCloseTo(Math.PI, 10);
    });
  });
});
