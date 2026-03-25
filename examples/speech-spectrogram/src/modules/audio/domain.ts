import type { Effect } from "effect";

export type AudioIoErrorCode =
  | "permission-denied"
  | "device-unavailable"
  | "unsupported-browser"
  | "recording-failed"
  | "decoding-failed"
  | "playback-failed"
  | "unknown";

export class AudioIoError extends Error {
  readonly _tag = "AudioIoError";
  readonly code: AudioIoErrorCode;
  readonly originalCause?: unknown;

  constructor(code: AudioIoErrorCode, message: string, originalCause?: unknown) {
    super(message);
    this.name = "AudioIoError";
    this.code = code;
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
