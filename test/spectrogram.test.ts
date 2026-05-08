import { describe, expect, it } from "vitest";

import { spectrogram, stft } from "../src/xform/stft.ts";

describe("spectrogram()", () => {
  it("derives dB frames from a signal", () => {
    const signal = Float64Array.from({ length: 512 }, (_, index) =>
      Math.sin((2 * Math.PI * index) / 32),
    );

    const result = spectrogram(signal, {
      fftSize: 128,
      hopSize: 32,
      sampleRate: 16_000,
      window: "hann",
      floorDb: -90,
    });

    expect(result.stft.frames.length).toBe(result.dbFrames.length);
    expect(result.floorDb).toBe(-90);
    expect(result.dbFrames[0]?.length).toBe(result.stft.frequencies.length);
  });

  it("accepts an existing STFT result without recomputing shape", () => {
    const signal = Float64Array.from({ length: 256 }, (_, index) =>
      Math.sin((2 * Math.PI * index) / 16),
    );
    const transformed = stft(signal, {
      fftSize: 64,
      hopSize: 16,
      sampleRate: 8_000,
      window: "hann",
    });

    const result = spectrogram(transformed, { floorDb: -100 });
    expect(result.stft).toBe(transformed);
    expect(result.dbFrames.length).toBe(transformed.frames.length);
  });
});
