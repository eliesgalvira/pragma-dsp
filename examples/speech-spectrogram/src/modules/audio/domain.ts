import type { Effect } from "effect";

export class AudioIoError extends Error {
  readonly _tag = "AudioIoError";
  readonly originalCause?: unknown;

  constructor(message: string, originalCause?: unknown) {
    super(message);
    this.name = "AudioIoError";
    this.originalCause = originalCause;
  }
}

export type AudioSamples = {
  readonly samples: Float32Array;
  readonly sampleRate: number;
};

export type AudioPreviewFrame = AudioSamples & {
  readonly elapsedMs: number;
  readonly level: number;
  readonly peakAmplitude: number;
};

export type AudioCaptureOptions = {
  readonly sampleRate: number;
  readonly previewFftSize: number;
  readonly previewWindowMs: number;
  readonly previewIntervalMs: number;
  readonly onFrame: (frame: AudioPreviewFrame) => void;
};

export type RecordingSession = {
  readonly stop: Effect.Effect<AudioSamples, AudioIoError>;
  readonly cancel: Effect.Effect<void, AudioIoError>;
};
