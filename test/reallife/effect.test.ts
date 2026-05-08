import { describe, expect, it } from "vitest";
import * as Effect from "effect/Effect";
import * as ManagedRuntime from "effect/ManagedRuntime";
import * as Stream from "effect/Stream";
import { spectrum } from "../../src/public/spectrum.ts";
import {
  Fourier,
  FourierLive,
  spectrumFx,
  spectrumStream
} from "../../src/effect/index.ts";
import { loadPureSine } from "./helpers.ts";

describe("Effect API parity", () => {
  const runtime = ManagedRuntime.make(FourierLive);

  describe("spectrumFx", () => {
    const ref = loadPureSine();

    for (const testCase of ref.cases.slice(0, 3)) {
      it(`spectrumFx matches spectrum() for ${testCase.name}`, () => {
        const options = {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: "rect" as const,
          sides: "one" as const
        };

        // Pure API result
        const pureResult = spectrum(testCase.signal, options);

        // Effect API result
        const effectResult = runtime.runSync(spectrumFx(testCase.signal, options));

        // Results should be identical
        expect(effectResult.peak.index).toBe(pureResult.peak.index);
        expect(effectResult.peak.frequency).toBe(pureResult.peak.frequency);
        expect(effectResult.peak.amplitude).toBe(pureResult.peak.amplitude);
        expect(effectResult.peak.phase).toBe(pureResult.peak.phase);

        // Arrays should be identical
        expect(effectResult.amplitude.length).toBe(pureResult.amplitude.length);
        for (let i = 0; i < effectResult.amplitude.length; i += 1) {
          expect(effectResult.amplitude[i]).toBe(pureResult.amplitude[i]);
          expect(effectResult.phase[i]).toBe(pureResult.phase[i]);
          expect(effectResult.frequencies[i]).toBe(pureResult.frequencies[i]);
        }
      });
    }
  });

  describe("Fourier service caching", () => {
    it("returns same FFT instance for same size", () => {
      const program = Effect.gen(function* () {
        const service = yield* Fourier;
        const fft1 = service.fft(64);
        const fft2 = service.fft(64);
        const fft3 = service.fft(128);

        return { fft1, fft2, fft3 };
      });

      const result = runtime.runSync(program);

      // Same size should return same instance
      expect(result.fft1).toBe(result.fft2);

      // Different size should return different instance
      expect(result.fft1).not.toBe(result.fft3);
    });

    it("returns same window instance for same type and size", () => {
      const program = Effect.gen(function* () {
        const service = yield* Fourier;
        const w1 = service.window("hann", 64);
        const w2 = service.window("hann", 64);
        const w3 = service.window("hamming", 64);
        const w4 = service.window("hann", 128);

        return { w1, w2, w3, w4 };
      });

      const result = runtime.runSync(program);

      // Same type and size should return same instance
      expect(result.w1).toBe(result.w2);

      // Different type should return different instance
      expect(result.w1).not.toBe(result.w3);

      // Different size should return different instance
      expect(result.w1).not.toBe(result.w4);
    });
  });

  describe("spectrumStream", () => {
    it("processes multiple frames correctly", () => {
      const ref = loadPureSine();
      const testCase = ref.cases[0];
      if (testCase === undefined) throw new Error("No test cases");

      const options = {
        sampleRate: testCase.sampleRate,
        fftSize: testCase.n,
        window: "rect" as const,
        sides: "one" as const
      };

      // Create a stream of 3 identical frames
      const frame = Float32Array.from(testCase.signal);
      const frames = Stream.fromIterable([frame, frame, frame]);

      // Process with spectrumStream
      const results = runtime.runSync(
        Stream.runCollect(spectrumStream(frames, options))
      );

      // Should have 3 results
      const resultsArray = Array.from(results);
      expect(resultsArray.length).toBe(3);

      // All results should be identical (same input frame)
      const expected = spectrum(testCase.signal, options);
      for (const result of resultsArray) {
        expect(result.peak.index).toBe(expected.peak.index);
        expect(result.peak.frequency).toBe(expected.peak.frequency);
        // Allow small floating point differences due to Float32Array conversion
        expect(result.peak.amplitude).toBeCloseTo(expected.peak.amplitude, 5);
      }
    });

    it("handles empty stream", () => {
      const frames = Stream.empty as Stream.Stream<Float32Array>;

      const results = runtime.runSync(
        Stream.runCollect(
          spectrumStream(frames, { sampleRate: 48000, fftSize: 64 })
        )
      );

      expect(Array.from(results).length).toBe(0);
    });
  });

  describe("window types in Effect API", () => {
    const windowTypes = ["rect", "hann", "hamming", "blackman"] as const;

    for (const windowType of windowTypes) {
      it(`supports ${windowType} window`, () => {
        const ref = loadPureSine();
        const testCase = ref.cases[0];
        if (testCase === undefined) throw new Error("No test cases");

        const options = {
          sampleRate: testCase.sampleRate,
          fftSize: testCase.n,
          window: windowType,
          sides: "one" as const
        };

        // Both APIs should work with all window types
        const pureResult = spectrum(testCase.signal, options);
        const effectResult = runtime.runSync(spectrumFx(testCase.signal, options));

        expect(effectResult.peak.index).toBe(pureResult.peak.index);
        expect(effectResult.peak.amplitude).toBeCloseTo(
          pureResult.peak.amplitude,
          10
        );
      });
    }
  });
});
