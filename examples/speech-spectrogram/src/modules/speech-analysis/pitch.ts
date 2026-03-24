import { spectralEnvelope } from "pragma-dsp/analysis";
import { autocorrelation } from "pragma-dsp/xform/fourier";

import type { FormantResult, PitchResult } from "./domain";

export function detectPitch(
  samples: Float32Array | Float64Array,
  sampleRate: number,
  opts: { readonly minF0?: number; readonly maxF0?: number } = {},
): PitchResult {
  const minF0 = opts.minF0 ?? 50;
  const maxF0 = opts.maxF0 ?? 600;
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
    f0: confidence > 0.2 ? sampleRate / refinedLag : null,
    confidence,
  };
}

export function detectFormants(
  magnitudes: Float64Array,
  frequencies: Float64Array,
  opts: {
    readonly smoothingWidth?: number;
    readonly maxFormants?: number;
    readonly minFreq?: number;
    readonly maxFreq?: number;
  } = {},
): FormantResult {
  const smoothingWidth = opts.smoothingWidth ?? 15;
  const maxFormants = opts.maxFormants ?? 4;
  const minFreq = opts.minFreq ?? 200;
  const maxFreq = opts.maxFreq ?? 5500;

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
}
