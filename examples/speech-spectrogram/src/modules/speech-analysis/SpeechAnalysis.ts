import { Effect, Layer, ServiceMap } from "effect";
import { analyzeSpeech } from "pragma-dsp/analysis";
import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";

import type { AudioSamples } from "../audio";
import type { AnalysisConfig, EditedSignal, SignalAnalysis, SpectralEditKind } from "./domain";

const ensureAnalysisWindow = (samples: Float32Array, fftSize: number) => {
  if (samples.length >= fftSize) {
    return samples;
  }

  const padded = new Float32Array(fftSize);
  padded.set(samples);
  return padded;
};

const analyzeInternal = (audio: AudioSamples, config: AnalysisConfig): SignalAnalysis => {
  const normalizedSamples = ensureAnalysisWindow(audio.samples, config.fftSize);
  const result = analyzeSpeech(normalizedSamples, {
    sampleRate: audio.sampleRate,
    fftSize: config.fftSize,
    hopSize: config.hopSize,
    window: "hann",
  });
  return result;
};

const trimEditedSignal = (edited: Float64Array, originalLength: number, edit: SpectralEditKind) => {
  const offset =
    edit.type === "complex_multiply" && edit.conjugate
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

const nextPowerOfTwo = (value: number) => {
  let power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
};

const applySpectralEdit = (samples: Float32Array, edit: SpectralEditKind) => {
  if (edit.type === "identity") {
    return Float64Array.from(samples);
  }

  const fftSize = nextPowerOfTwo(samples.length);
  const fft = new FluentFFT(fftSize);
  const input = new Float64Array(fftSize);
  input.set(samples);
  const chain = fft.forward(input).mulScalar(edit.real, edit.imaginary);
  if (edit.conjugate) {
    chain.conj();
  }
  return chain.inverse().real;
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
          const edited = trimEditedSignal(
            applySpectralEdit(audio.samples, edit),
            audio.samples.length,
            edit,
          );
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
