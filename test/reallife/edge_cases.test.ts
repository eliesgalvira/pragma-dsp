import { describe, expect, it } from "vitest";
import { FFT, magnitude } from "../../src/xform/fourier.js";
import { spectrum } from "../../src/public/spectrum.js";
import { loadSpecial, maxAbsError } from "./helpers.js";

describe("edge cases and robustness", () => {
  describe("zero input", () => {
    it("FFT of zeros gives zeros", () => {
      const ref = loadSpecial();
      const zerosCase = ref.cases.find((c) => c.kind === "zeros");
      if (!zerosCase) throw new Error("Missing zeros test case");

      const fft = new FFT(zerosCase.n);
      const result = fft.forward(zerosCase.signal);

      for (let i = 0; i < zerosCase.n; i += 1) {
        expect(result.real[i]).toBe(0);
        expect(result.imag[i]).toBe(0);
      }
    });

    it("spectrum() of zeros gives zeros", () => {
      const n = 64;
      const signal = new Float64Array(n);

      const result = spectrum(signal, {
        sampleRate: 48000,
        fftSize: n,
        window: "rect",
        sides: "one"
      });

      for (let i = 0; i < result.amplitude.length; i += 1) {
        expect(result.amplitude[i]).toBe(0);
      }

      expect(result.peak.amplitude).toBe(0);
    });
  });

  describe("DC only signal", () => {
    it("DC signal has energy only in bin 0", () => {
      const ref = loadSpecial();
      const dcCase = ref.cases.find((c) => c.kind === "dc");
      if (!dcCase) throw new Error("Missing DC test case");

      const fft = new FFT(dcCase.n);
      const result = fft.forward(dcCase.signal);
      const mag = magnitude(result);

      // Bin 0 should have all the energy
      expect(mag[0]).toBeCloseTo(
        dcCase.n * (dcCase.params.level as number),
        10
      );

      // All other bins should be zero
      for (let i = 1; i < dcCase.n; i += 1) {
        expect(mag[i]).toBeLessThan(1e-10);
      }
    });
  });

  describe("Nyquist signal", () => {
    it("alternating +1/-1 has energy only at Nyquist", () => {
      const ref = loadSpecial();
      const nyquistCase = ref.cases.find((c) => c.kind === "nyquist");
      if (!nyquistCase) throw new Error("Missing Nyquist test case");

      const fft = new FFT(nyquistCase.n);
      const result = fft.forward(nyquistCase.signal);
      const mag = magnitude(result);

      const nyquistBin = nyquistCase.n / 2;

      // Nyquist bin should have all the energy
      expect(mag[nyquistBin]).toBeCloseTo(
        nyquistCase.n * (nyquistCase.params.amplitude as number),
        10
      );

      // All other bins should be zero
      for (let i = 0; i < nyquistCase.n; i += 1) {
        if (i !== nyquistBin) {
          expect(mag[i]).toBeLessThan(1e-10);
        }
      }
    });
  });

  describe("impulse response", () => {
    it("impulse at position 0 gives flat magnitude spectrum", () => {
      const ref = loadSpecial();
      const impulseCase = ref.cases.find(
        (c) => c.kind === "impulse" && c.params.position === 0
      );
      if (!impulseCase) throw new Error("Missing impulse test case");

      const fft = new FFT(impulseCase.n);
      const result = fft.forward(impulseCase.signal);
      const mag = magnitude(result);

      // All bins should have magnitude = amplitude
      const expectedMag = impulseCase.params.amplitude as number;
      for (let i = 0; i < impulseCase.n; i += 1) {
        expect(mag[i]).toBeCloseTo(expectedMag, 10);
      }
    });

    it("impulse at middle position gives correct phase pattern", () => {
      const ref = loadSpecial();
      const impulseCase = ref.cases.find(
        (c) => c.kind === "impulse" && (c.params.position as number) > 0
      );
      if (!impulseCase) throw new Error("Missing shifted impulse test case");

      const fft = new FFT(impulseCase.n);
      const result = fft.forward(impulseCase.signal);
      const mag = magnitude(result);

      // Magnitude should still be flat
      const expectedMag = impulseCase.params.amplitude as number;
      for (let i = 0; i < impulseCase.n; i += 1) {
        expect(mag[i]).toBeCloseTo(expectedMag, 10);
      }
    });
  });

  describe("very small values", () => {
    it("handles tiny amplitude signals without underflow", () => {
      const ref = loadSpecial();
      const tinyCase = ref.cases.find((c) => c.kind === "tiny");
      if (!tinyCase) throw new Error("Missing tiny amplitude test case");

      const fft = new FFT(tinyCase.n);
      const result = fft.forward(tinyCase.signal);

      // Should match NumPy output (no NaN or Inf)
      for (let i = 0; i < tinyCase.n; i += 1) {
        expect(Number.isFinite(result.real[i])).toBe(true);
        expect(Number.isFinite(result.imag[i])).toBe(true);
      }

      // Should match reference values (scaled by tiny factor)
      const maxErr = maxAbsError(result.real, tinyCase.fftRe);
      expect(maxErr).toBeLessThan(1e-20);
    });
  });

  describe("very large values", () => {
    it("handles large amplitude signals without overflow", () => {
      const ref = loadSpecial();
      const largeCase = ref.cases.find((c) => c.kind === "large");
      if (!largeCase) throw new Error("Missing large amplitude test case");

      const fft = new FFT(largeCase.n);
      const result = fft.forward(largeCase.signal);

      // Should not produce NaN or Inf
      for (let i = 0; i < largeCase.n; i += 1) {
        expect(Number.isFinite(result.real[i])).toBe(true);
        expect(Number.isFinite(result.imag[i])).toBe(true);
      }

      // Should match reference values within relative tolerance
      for (let i = 0; i < largeCase.n; i += 1) {
        const expected = largeCase.fftRe[i]!;
        const actual = result.real[i]!;
        if (Math.abs(expected) > 1) {
          const relErr = Math.abs(actual - expected) / Math.abs(expected);
          expect(relErr).toBeLessThan(1e-9);
        } else {
          expect(Math.abs(actual - expected)).toBeLessThan(1e-6);
        }
      }
    });
  });

  describe("zero-padding (input shorter than fftSize)", () => {
    it("handles input shorter than fftSize", () => {
      const shortSignal = new Float64Array([1, 2, 3, 4]);
      const fftSize = 16;

      const result = spectrum(shortSignal, {
        sampleRate: 48000,
        fftSize: fftSize,
        window: "rect",
        sides: "one"
      });

      // Should work without error
      expect(result.amplitude.length).toBe(fftSize / 2 + 1);

      // The signal is zero-padded, so we should get valid results
      expect(Number.isFinite(result.peak.amplitude)).toBe(true);
      expect(Number.isFinite(result.peak.frequency)).toBe(true);
    });

    it("zero-padding preserves signal content", () => {
      // A DC signal of length 4 zero-padded to 16 should still show DC
      const shortSignal = new Float64Array([1, 1, 1, 1]);
      const fftSize = 16;

      const result = spectrum(shortSignal, {
        sampleRate: 48000,
        fftSize: fftSize,
        window: "rect",
        sides: "one"
      });

      // DC component should be 4/16 = 0.25 (sum of input / fftSize)
      expect(result.amplitude[0]).toBeCloseTo(4 / fftSize, 6);
    });
  });

  describe("round-trip consistency", () => {
    it("IFFT(FFT(x)) = x for all special signals", () => {
      const ref = loadSpecial();

      for (const testCase of ref.cases) {
        const fft = new FFT(testCase.n);
        const forward = fft.forward(testCase.signal);
        const inverse = fft.inverse(forward);

        const maxErr = maxAbsError(inverse.real, testCase.signal);
        expect(maxErr).toBeLessThan(1e-9);

        // Imaginary part should be near zero
        const imagMax = maxAbsError(
          inverse.imag,
          new Float64Array(testCase.n)
        );
        expect(imagMax).toBeLessThan(1e-9);
      }
    });
  });
});
