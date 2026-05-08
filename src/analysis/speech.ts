import type { StftResult } from "../xform/stft.ts";
import { stft } from "../xform/stft.ts";
import { autocorrelation } from "../xform/fourier.ts";
import { spectralEnvelope } from "./spectral.ts";

export type PitchResult = {
  readonly f0: number | null;
  readonly confidence: number;
};

export type FormantResult = {
  readonly formants: number[];
  readonly envelope: Float64Array;
};

export type DetectPitchOptions = {
  readonly minF0?: number;
  readonly maxF0?: number;
  readonly confidenceThreshold?: number;
};

export type DetectFormantsOptions = {
  readonly smoothingWidth?: number;
  readonly maxFormants?: number;
  readonly minFreq?: number;
  readonly maxFreq?: number;
};

export type AnalyzeSpeechOptions = {
  readonly sampleRate: number;
  readonly fftSize?: number;
  readonly hopSize?: number;
  readonly window?: "rect" | "hann" | "hamming" | "blackman";
  readonly pitch?: DetectPitchOptions;
  readonly formants?: DetectFormantsOptions;
};

export type SpeechAnalysisResult = {
  readonly stft: StftResult;
  readonly pitchTrack: ReadonlyArray<number | null>;
  readonly formants: ReadonlyArray<FormantResult>;
  readonly medianF0: number | null;
  readonly formantMedians: ReadonlyArray<number>;
};

const median = (values: ReadonlyArray<number>) => {
  if (values.length === 0) {
    return null;
  }

  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
};

const ensureFrameLength = (samples: ArrayLike<number>, fftSize: number) => {
  if (samples.length >= fftSize) {
    return Float64Array.from(samples);
  }

  const padded = new Float64Array(fftSize);
  for (let index = 0; index < samples.length; index++) {
    padded[index] = samples[index] ?? 0;
  }
  return padded;
};

const createFrameSlices = (
  samples: ArrayLike<number>,
  fftSize: number,
  hopSize: number,
) => {
  if (samples.length <= fftSize) {
    return [ensureFrameLength(samples, fftSize)];
  }

  const frames: Array<Float64Array> = [];
  for (let start = 0; start + fftSize <= samples.length; start += hopSize) {
    const frame = new Float64Array(fftSize);
    for (let index = 0; index < fftSize; index++) {
      frame[index] = samples[start + index] ?? 0;
    }
    frames.push(frame);
  }
  return frames;
};

export const detectPitch = (
  samples: ArrayLike<number>,
  sampleRate: number,
  options: DetectPitchOptions = {},
): PitchResult => {
  const minF0 = options.minF0 ?? 50;
  const maxF0 = options.maxF0 ?? 600;
  const confidenceThreshold = options.confidenceThreshold ?? 0.2;
  const correlation = autocorrelation(samples);
  const zeroLag = correlation[0] ?? 0;

  if (zeroLag === 0) {
    return { f0: null, confidence: 0 };
  }

  const minLag = Math.floor(sampleRate / maxF0);
  const maxLag = Math.min(Math.ceil(sampleRate / minF0), samples.length - 1);
  let bestLag = minLag;
  let bestValue = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    const value = correlation[lag] ?? -Infinity;
    if (value > bestValue) {
      bestValue = value;
      bestLag = lag;
    }
  }

  const previous = bestLag > 0 ? (correlation[bestLag - 1] ?? bestValue) : bestValue;
  const next = bestLag < correlation.length - 1 ? (correlation[bestLag + 1] ?? bestValue) : bestValue;
  const denominator = previous - 2 * bestValue + next;
  const correction = denominator === 0 ? 0 : (previous - next) / (2 * denominator);
  const refinedLag = bestLag + correction;
  const confidence = bestValue / zeroLag;

  return {
    f0: confidence > confidenceThreshold ? sampleRate / refinedLag : null,
    confidence,
  };
};

export const detectFormants = (
  magnitudes: ArrayLike<number>,
  frequencies: ArrayLike<number>,
  options: DetectFormantsOptions = {},
): FormantResult => {
  const smoothingWidth = options.smoothingWidth ?? 15;
  const maxFormants = options.maxFormants ?? 4;
  const minFreq = options.minFreq ?? 200;
  const maxFreq = options.maxFreq ?? 5500;
  const envelope = spectralEnvelope(magnitudes, { smoothingWidth });
  const peaks: Array<{ readonly freq: number; readonly magnitude: number }> = [];

  for (let index = 1; index < envelope.length - 1; index++) {
    const frequency = frequencies[index] ?? 0;
    if (frequency < minFreq || frequency > maxFreq) {
      continue;
    }

    const previous = envelope[index - 1] ?? 0;
    const current = envelope[index] ?? 0;
    const next = envelope[index + 1] ?? 0;

    if (current > 0 && current > previous && current > next) {
      peaks.push({ freq: frequency, magnitude: current });
    }
  }

  peaks.sort((left, right) => right.magnitude - left.magnitude);
  const selected = peaks.slice(0, maxFormants).sort((left, right) => left.freq - right.freq);

  return {
    formants: selected.map((peak) => peak.freq),
    envelope,
  };
};

export const analyzeSpeech = (
  samples: ArrayLike<number>,
  options: AnalyzeSpeechOptions,
): SpeechAnalysisResult => {
  const fftSize = options.fftSize ?? 1024;
  const hopSize = options.hopSize ?? Math.floor(fftSize / 4);
  const sampleRate = options.sampleRate;
  const prepared = ensureFrameLength(samples, fftSize);
  const stftResult = stft(prepared, {
    fftSize,
    hopSize,
    sampleRate,
    window: options.window ?? "hann",
    complexSides: "one",
  });

  const frames = createFrameSlices(prepared, fftSize, hopSize);
  const pitchTrack = frames.map((frame) => detectPitch(frame, sampleRate, options.pitch).f0);
  const formants = stftResult.frames.map((frame) =>
    detectFormants(frame.magnitudes, stftResult.frequencies, options.formants),
  );

  const medianF0 = median(pitchTrack.filter((value): value is number => value != null && value > 0));
  const formantMedians = Array.from({ length: 4 }, (_, formantIndex) =>
    median(
      formants
        .map((frame) => frame.formants[formantIndex])
        .filter((value): value is number => value != null && value > 0),
    ),
  ).filter((value): value is number => value != null);

  return {
    stft: stftResult,
    pitchTrack,
    formants,
    medianF0,
    formantMedians,
  };
};
