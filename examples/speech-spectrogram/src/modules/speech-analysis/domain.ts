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
  | { readonly type: "scale"; readonly factor: number }
  | { readonly type: "multiply_by_i" }
  | { readonly type: "conjugate" }
  | { readonly type: "negate" }
  | { readonly type: "scale_and_conjugate"; readonly factor: number };

export type SpectralEditPreset = {
  readonly id: string;
  readonly label: string;
  readonly edit: SpectralEditKind;
};

export const SPECTRAL_EDIT_PRESETS: ReadonlyArray<SpectralEditPreset> = [
  { id: "original", label: "Original", edit: { type: "identity" } },
  { id: "scale-x4", label: "Scale x4", edit: { type: "scale", factor: 4 } },
  { id: "scale-x025", label: "Scale x0.25", edit: { type: "scale", factor: 0.25 } },
  { id: "multiply-by-i", label: "Multiply by i", edit: { type: "multiply_by_i" } },
  { id: "conjugate", label: "Conjugate", edit: { type: "conjugate" } },
  { id: "negate", label: "Negate", edit: { type: "negate" } },
  {
    id: "scale-x2-conjugate",
    label: "Scale x2 + Conjugate",
    edit: { type: "scale_and_conjugate", factor: 2 },
  },
];

export const DEFAULT_SPECTRAL_EDIT = SPECTRAL_EDIT_PRESETS[0]!.edit;

export type EditedSignal = {
  readonly audio: AudioSamples;
  readonly difference: Float32Array;
  readonly analysis: SignalAnalysis;
};
