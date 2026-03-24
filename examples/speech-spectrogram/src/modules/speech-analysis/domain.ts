import type { StftResult } from "pragma-dsp/xform/stft";

import type { AudioSamples } from "../audio";

export type AnalysisConfig = {
  readonly fftSize: number;
  readonly hopSize: number;
  readonly previewWindowMs: number;
  readonly previewIntervalMs: number;
  readonly previewFftSize: number;
};

export const DEFAULT_ANALYSIS_CONFIG: AnalysisConfig = {
  fftSize: 1024,
  hopSize: 256,
  previewWindowMs: 1400,
  previewIntervalMs: 120,
  previewFftSize: 2048,
};

export type PitchResult = {
  readonly f0: number | null;
  readonly confidence: number;
};

export type FormantResult = {
  readonly formants: number[];
  readonly envelope: Float64Array;
};

export type SignalAnalysis = {
  readonly stft: StftResult;
  readonly pitchTrack: ReadonlyArray<number | null>;
  readonly formants: ReadonlyArray<FormantResult>;
  readonly medianF0: number | null;
  readonly formantMedians: ReadonlyArray<number>;
};

export type SpectralEditKind =
  | { readonly type: "identity" }
  | { readonly type: "complex_multiply"; readonly real: number; readonly imaginary: number };

export const DEFAULT_SPECTRAL_EDIT: SpectralEditKind = { type: "identity" };

export const spectralEditEquals = (left: SpectralEditKind, right: SpectralEditKind) => {
  if (left.type === "identity" && right.type === "identity") {
    return true;
  }

  if (left.type !== "complex_multiply" || right.type !== "complex_multiply") {
    return false;
  }

  return left.real === right.real && left.imaginary === right.imaginary;
};

export const formatSpectralEditLabel = (edit: SpectralEditKind) => {
  if (edit.type === "identity") {
    return "Original";
  }

  const sign = edit.imaginary < 0 ? "-" : "+";
  return `${edit.real.toFixed(2)} ${sign} ${Math.abs(edit.imaginary).toFixed(2)}i`;
};

export type EditedSignal = {
  readonly audio: AudioSamples;
  readonly difference: Float32Array;
  readonly analysis: SignalAnalysis;
};
