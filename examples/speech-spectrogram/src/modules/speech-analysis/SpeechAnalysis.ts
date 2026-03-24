import { Effect, Layer, ServiceMap } from "effect";
import { stft } from "pragma-dsp/xform/stft";

import type { AudioSamples } from "../audio";
import type {
  AnalysisConfig,
  EditedSignal,
  SignalAnalysis,
  SpectralEditKind,
} from "./domain";
import { applySpectralEdit } from "./spectralEdit";
import { detectFormants, detectPitch } from "./pitch";

const median = (values: ReadonlyArray<number>) => {
  if (values.length === 0) {
    return null;
  }

  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
};

const ensureAnalysisWindow = (samples: Float32Array, fftSize: number) => {
  if (samples.length >= fftSize) {
    return samples;
  }

  const padded = new Float32Array(fftSize);
  padded.set(samples);
  return padded;
};

const createFrameSlices = (samples: Float32Array, fftSize: number, hopSize: number) => {
  if (samples.length <= fftSize) {
    return [ensureAnalysisWindow(samples, fftSize)];
  }

  const frames: Array<Float32Array> = [];
  for (let start = 0; start + fftSize <= samples.length; start += hopSize) {
    frames.push(samples.slice(start, start + fftSize));
  }
  return frames;
};

const analyzeInternal = (audio: AudioSamples, config: AnalysisConfig): SignalAnalysis => {
  const normalizedSamples = ensureAnalysisWindow(audio.samples, config.fftSize);
  const stftResult = stft(normalizedSamples, {
    fftSize: config.fftSize,
    hopSize: config.hopSize,
    sampleRate: audio.sampleRate,
    window: "hann",
    complexSides: "one",
  });

  const frameSlices = createFrameSlices(normalizedSamples, config.fftSize, config.hopSize);
  const pitchTrack = frameSlices.map((frame) => detectPitch(frame, audio.sampleRate).f0);
  const formants = stftResult.frames.map((frame) =>
    detectFormants(frame.magnitudes, stftResult.frequencies),
  );

  const medianF0 = median(pitchTrack.filter((pitch): pitch is number => pitch != null && pitch > 0));
  const formantMedians = Array.from({ length: 4 }, (_, formantIndex) =>
    median(
      formants
        .map((frame) => frame.formants[formantIndex])
        .filter((frequency): frequency is number => frequency != null && frequency > 0),
    ),
  ).filter((frequency): frequency is number => frequency != null);

  return {
    stft: stftResult,
    pitchTrack,
    formants,
    medianF0,
    formantMedians,
  };
};

const trimEditedSignal = (edited: Float64Array, originalLength: number, edit: SpectralEditKind) => {
  const offset =
    edit.type === "conjugate" || edit.type === "scale_and_conjugate"
      ? edited.length - originalLength
      : 0;

  const signal = new Float32Array(originalLength);
  for (let index = 0; index < originalLength; index++) {
    signal[index] = edited[offset + index] ?? 0;
  }
  return signal;
};

const difference = (left: Float32Array, right: Float32Array) => {
  const output = new Float32Array(left.length);
  for (let index = 0; index < left.length; index++) {
    output[index] = (left[index] ?? 0) - (right[index] ?? 0);
  }
  return output;
};

export class SpeechAnalysis extends ServiceMap.Service<
  SpeechAnalysis,
  {
    readonly analyzeSignal: (
      audio: AudioSamples,
      config: AnalysisConfig,
    ) => Effect.Effect<SignalAnalysis>;
    readonly applyEdit: (
      audio: AudioSamples,
      edit: SpectralEditKind,
      config: AnalysisConfig,
    ) => Effect.Effect<EditedSignal>;
  }
>()("@speech/analysis/SpeechAnalysis") {
  static readonly layer = Layer.succeed(this)({
    analyzeSignal: Effect.fn("SpeechAnalysis.analyzeSignal")((audio: AudioSamples, config: AnalysisConfig) =>
      Effect.sync(() => analyzeInternal(audio, config)),
    ),
    applyEdit: Effect.fn("SpeechAnalysis.applyEdit")(
      (audio: AudioSamples, edit: SpectralEditKind, config: AnalysisConfig) =>
        Effect.sync(() => {
          const edited = trimEditedSignal(applySpectralEdit(audio.samples, edit), audio.samples.length, edit);
          const editedAudio: AudioSamples = { samples: edited, sampleRate: audio.sampleRate };
          return {
            audio: editedAudio,
            difference: difference(audio.samples, edited),
            analysis: analyzeInternal(editedAudio, config),
          };
        }),
    ),
  });
}
