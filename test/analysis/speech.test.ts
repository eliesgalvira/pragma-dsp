import { describe, expect, it } from "vitest";

import { analyzeSpeech, detectFormants, detectPitch } from "../../src/analysis/index.js";

describe("speech analysis helpers", () => {
  it("detectPitch finds a bin-centered sine", () => {
    const sampleRate = 16_000;
    const frequency = 200;
    const size = 1024;
    const samples = Float64Array.from({ length: size }, (_, index) =>
      Math.sin((2 * Math.PI * frequency * index) / sampleRate),
    );

    const result = detectPitch(samples, sampleRate);
    expect(result.f0).not.toBeNull();
    expect(result.f0!).toBeCloseTo(frequency, 0);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it("detectFormants returns ordered peaks within the configured band", () => {
    const frequencies = Float64Array.from([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]);
    const magnitudes = Float64Array.from([0.2, 0.8, 0.5, 1.4, 0.6, 1.2, 0.5, 0.9, 0.2]);

    const result = detectFormants(magnitudes, frequencies, {
      smoothingWidth: 1,
      minFreq: 200,
      maxFreq: 1800,
      maxFormants: 3,
    });

    expect(result.formants.length).toBeGreaterThan(0);
    expect(result.formants).toEqual([...result.formants].sort((a, b) => a - b));
  });

  it("analyzeSpeech returns aligned STFT, pitch track, and formant frames", () => {
    const sampleRate = 16_000;
    const fftSize = 256;
    const hopSize = 64;
    const samples = Float64Array.from({ length: 1024 }, (_, index) =>
      Math.sin((2 * Math.PI * 220 * index) / sampleRate),
    );

    const result = analyzeSpeech(samples, {
      sampleRate,
      fftSize,
      hopSize,
      window: "hann",
    });

    expect(result.stft.frames.length).toBeGreaterThan(0);
    expect(result.pitchTrack.length).toBe(result.stft.frames.length);
    expect(result.formants.length).toBe(result.stft.frames.length);
    expect(result.medianF0).not.toBeNull();
  });
});
