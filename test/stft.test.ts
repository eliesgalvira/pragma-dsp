import { describe, expect, it } from "vitest";
import { istft, stft } from "../src/xform/stft.ts";

describe("stft()", () => {
  it("produces expected frame, bin, and time shapes", () => {
    const samples = Float64Array.from([1, 0, -1, 0, 1, 0, -1, 0]);
    const result = stft(samples, {
      fftSize: 4,
      hopSize: 2,
      sampleRate: 8,
      window: "rect",
      sides: "one",
    });

    expect(result.frames.length).toBe(3);
    expect(result.frequencies.length).toBe(3);
    expect(result.times.length).toBe(3);
    expect(result.fftSize).toBe(4);
    expect(result.hopSize).toBe(2);
    expect(result.sampleRate).toBe(8);

    for (const frame of result.frames) {
      expect(frame.complex.real.length).toBe(4);
      expect(frame.magnitudes.length).toBe(3);
      expect(frame.phases.length).toBe(3);
    }
  });

  it("supports one-sided complex storage", () => {
    const samples = Float64Array.from([1, 0, -1, 0, 1, 0, -1, 0]);
    const result = stft(samples, {
      fftSize: 4,
      hopSize: 2,
      sampleRate: 8,
      window: "rect",
      sides: "one",
      complexSides: "one",
    });

    for (const frame of result.frames) {
      expect(frame.complex.real.length).toBe(3);
      expect(frame.magnitudes.length).toBe(3);
    }
  });

  it("round-trips one-sided complex via inverse()", () => {
    const samples = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = stft(samples, {
      fftSize: 4,
      hopSize: 4,
      sampleRate: 8,
      window: "rect",
      sides: "one",
      complexSides: "one",
    });

    const reconstructed = result.inverse({ outputLength: samples.length });
    for (let i = 0; i < samples.length; i += 1) {
      expect(reconstructed[i]).toBeCloseTo(samples[i]!, 8);
    }
  });

  it("standalone istft matches chained inverse", () => {
    const samples = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = stft(samples, {
      fftSize: 4,
      hopSize: 4,
      sampleRate: 8,
      window: "rect",
      sides: "one",
      complexSides: "one",
    });

    const chained = result.inverse({ outputLength: samples.length });
    const standalone = istft(result.frames, {
      fftSize: result.fftSize,
      hopSize: result.hopSize,
      window: result.window,
      complexSides: result.complexSides,
      outputLength: samples.length,
    });

    for (let i = 0; i < samples.length; i += 1) {
      expect(standalone[i]).toBeCloseTo(chained[i]!, 10);
    }
  });
});
