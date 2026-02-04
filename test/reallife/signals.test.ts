import { describe, expect, it } from "vitest";
import { FFT, magnitude, phase } from "../../src/xform/fourier.js";
import {
  expectCloseArray,
  loadChirp,
  loadMultiTone,
  loadPureSine,
  loadSpecial,
  maxAbsError
} from "./helpers.js";

describe("FFT correctness vs NumPy", () => {
  describe("pure sine waves", () => {
    const ref = loadPureSine();

    for (const testCase of ref.cases) {
      it(`matches NumPy FFT for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);

        // Check real and imaginary parts match NumPy
        expectCloseArray(result.real, testCase.fftRe, 1e-10);
        expectCloseArray(result.imag, testCase.fftIm, 1e-10);
      });

      it(`magnitude matches NumPy for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);
        const mag = magnitude(result);

        expectCloseArray(mag, testCase.magnitude, 1e-10);
      });

      it(`phase matches NumPy for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);
        const ph = phase(result);

        // Phase comparison is tricky for bins with near-zero magnitude
        // Only check bins with significant magnitude
        for (let i = 0; i < testCase.n; i += 1) {
          if (testCase.magnitude[i]! > 1e-6) {
            const diff = Math.abs(ph[i]! - testCase.phase[i]!);
            // Handle phase wrapping (allow 2*pi difference)
            const wrappedDiff = Math.min(diff, Math.abs(diff - 2 * Math.PI));
            expect(wrappedDiff).toBeLessThan(1e-10);
          }
        }
      });

      it(`round-trips correctly for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const forward = fft.forward(testCase.signal);
        const inverse = fft.inverse(forward);

        expectCloseArray(inverse.real, testCase.signal, 1e-10);
        // Imaginary part should be near zero for real input
        const imagMax = maxAbsError(
          inverse.imag,
          new Float64Array(testCase.n)
        );
        expect(imagMax).toBeLessThan(1e-10);
      });
    }
  });

  describe("multi-tone signals", () => {
    const ref = loadMultiTone();

    for (const testCase of ref.cases) {
      it(`matches NumPy FFT for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);

        expectCloseArray(result.real, testCase.fftRe, 1e-10);
        expectCloseArray(result.imag, testCase.fftIm, 1e-10);
      });

      it(`detects correct peaks for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);
        const mag = magnitude(result);

        const binIndices = testCase.params.bin_indices as number[];
        const amplitudes = testCase.params.amplitudes as number[];

        // Check that peaks appear at expected bins
        for (let i = 0; i < binIndices.length; i += 1) {
          const bin = binIndices[i]!;
          const expectedAmp = amplitudes[i]!;
          // For raw FFT, magnitude at bin k should be approximately N * amp / 2
          // (for a sine wave that's split between positive and negative frequency)
          const expectedMag = (testCase.n * expectedAmp) / 2;
          expect(mag[bin]).toBeCloseTo(expectedMag, 5);
        }
      });
    }
  });

  describe("chirp signals", () => {
    const ref = loadChirp();

    for (const testCase of ref.cases) {
      it(`matches NumPy FFT for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const result = fft.forward(testCase.signal);

        expectCloseArray(result.real, testCase.fftRe, 1e-10);
        expectCloseArray(result.imag, testCase.fftIm, 1e-10);
      });

      it(`round-trips correctly for ${testCase.name}`, () => {
        const fft = new FFT(testCase.n);
        const forward = fft.forward(testCase.signal);
        const inverse = fft.inverse(forward);

        expectCloseArray(inverse.real, testCase.signal, 1e-10);
      });
    }
  });

  describe("special signals", () => {
    const ref = loadSpecial();

    const impulseCase = ref.cases.find((c) => c.kind === "impulse");
    if (impulseCase) {
      it("impulse has flat magnitude spectrum", () => {
        const fft = new FFT(impulseCase.n);
        const result = fft.forward(impulseCase.signal);
        const mag = magnitude(result);

        // For impulse at position 0 with amplitude 1, all bins have magnitude 1
        if (impulseCase.params.position === 0) {
          for (let i = 0; i < impulseCase.n; i += 1) {
            expect(mag[i]).toBeCloseTo(impulseCase.params.amplitude as number, 10);
          }
        }
      });
    }

    const dcCase = ref.cases.find((c) => c.kind === "dc");
    if (dcCase) {
      it("DC signal has energy only in bin 0", () => {
        const fft = new FFT(dcCase.n);
        const result = fft.forward(dcCase.signal);
        const mag = magnitude(result);

        // Bin 0 should have magnitude = N * level
        expect(mag[0]).toBeCloseTo(
          dcCase.n * (dcCase.params.level as number),
          10
        );

        // Other bins should be zero
        for (let i = 1; i < dcCase.n; i += 1) {
          expect(mag[i]).toBeLessThan(1e-10);
        }
      });
    }

    const nyquistCase = ref.cases.find((c) => c.kind === "nyquist");
    if (nyquistCase) {
      it("Nyquist signal has energy only at Nyquist bin", () => {
        const fft = new FFT(nyquistCase.n);
        const result = fft.forward(nyquistCase.signal);
        const mag = magnitude(result);

        const nyquistBin = nyquistCase.n / 2;
        // Nyquist bin should have magnitude = N * amplitude
        expect(mag[nyquistBin]).toBeCloseTo(
          nyquistCase.n * (nyquistCase.params.amplitude as number),
          10
        );

        // Other bins should be zero (except possibly numerical noise)
        for (let i = 0; i < nyquistCase.n; i += 1) {
          if (i !== nyquistBin) {
            expect(mag[i]).toBeLessThan(1e-10);
          }
        }
      });
    }

    const zerosCase = ref.cases.find((c) => c.kind === "zeros");
    if (zerosCase) {
      it("zero input gives zero output", () => {
        const fft = new FFT(zerosCase.n);
        const result = fft.forward(zerosCase.signal);

        for (let i = 0; i < zerosCase.n; i += 1) {
          expect(result.real[i]).toBe(0);
          expect(result.imag[i]).toBe(0);
        }
      });
    }
  });
});
