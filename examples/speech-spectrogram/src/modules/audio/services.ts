import { Effect, Layer, ServiceMap } from "effect";

import {
  AudioIoError,
} from "./domain";
import type {
  AudioCaptureOptions,
  AudioPreviewFrame,
  AudioSamples,
  RecordingSession,
} from "./domain";

const toAudioIoError = (error: unknown): AudioIoError => {
  const message = error instanceof Error ? error.message : String(error);
  return new AudioIoError(message, error);
};

class RollingBuffer {
  private readonly data: Float32Array;
  private readonly capacity: number;
  private writeIndex = 0;
  private size = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.data = new Float32Array(capacity);
  }

  append(chunk: Float32Array) {
    if (chunk.length >= this.capacity) {
      this.data.set(chunk.subarray(chunk.length - this.capacity));
      this.writeIndex = 0;
      this.size = this.capacity;
      return;
    }

    for (let index = 0; index < chunk.length; index++) {
      this.data[this.writeIndex] = chunk[index] ?? 0;
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
    }

    this.size = Math.min(this.capacity, this.size + chunk.length);
  }

  snapshot() {
    if (this.size === 0) {
      return new Float32Array(0);
    }

    if (this.size < this.capacity) {
      return this.data.slice(0, this.size);
    }

    const output = new Float32Array(this.capacity);
    output.set(this.data.subarray(this.writeIndex));
    output.set(this.data.subarray(0, this.writeIndex), this.capacity - this.writeIndex);
    return output;
  }
}

const computeLevel = (samples: Float32Array) => {
  if (samples.length === 0) {
    return 0;
  }

  let sum = 0;
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index] ?? 0;
    sum += sample * sample;
  }
  return Math.sqrt(sum / samples.length);
};

const peakAmplitude = (samples: Float32Array) => {
  let peak = 0;
  for (let index = 0; index < samples.length; index++) {
    peak = Math.max(peak, Math.abs(samples[index] ?? 0));
  }
  return peak;
};

const decodeAudioBlob = async (blob: Blob, preferredSampleRate: number): Promise<AudioSamples> => {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();

  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const sampleRate = preferredSampleRate || decoded.sampleRate;
    const channelData = decoded.getChannelData(0);

    if (decoded.sampleRate === sampleRate) {
      const copied = new Float32Array(channelData.length);
      copied.set(channelData);
      return { samples: copied, sampleRate };
    }

    const durationSeconds = decoded.length / decoded.sampleRate;
    const resampledLength = Math.max(1, Math.ceil(durationSeconds * sampleRate));
    const offline = new OfflineAudioContext(decoded.numberOfChannels, resampledLength, sampleRate);
    const source = offline.createBufferSource();
    const buffer = offline.createBuffer(decoded.numberOfChannels, decoded.length, decoded.sampleRate);

    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const copiedChannel = new Float32Array(decoded.getChannelData(channel));
      buffer.copyToChannel(copiedChannel, channel);
    }

    source.buffer = buffer;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    const renderedChannel = rendered.getChannelData(0);
    const copied = new Float32Array(renderedChannel.length);
    copied.set(renderedChannel);
    return { samples: copied, sampleRate };
  } finally {
    await audioContext.close().catch(() => undefined);
  }
};

const createStopPromise = (
  recorder: MediaRecorder,
  chunks: Array<BlobPart>,
): Promise<Blob> =>
  new Promise((resolve, reject) => {
    const handleStop = () => {
      cleanup();
      resolve(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
    };

    const handleError = (event: Event) => {
      cleanup();
      const recorderEvent = event as Event & { readonly error?: DOMException };
      reject(recorderEvent.error ?? new Error("MediaRecorder failed"));
    };

    const cleanup = () => {
      recorder.removeEventListener("stop", handleStop);
      recorder.removeEventListener("error", handleError);
    };

    recorder.addEventListener("stop", handleStop, { once: true });
    recorder.addEventListener("error", handleError, { once: true });

    if (recorder.state === "inactive") {
      handleStop();
      return;
    }

    recorder.stop();
  });

export class AudioInput extends ServiceMap.Service<
  AudioInput,
  {
    readonly startRecording: (
      options: AudioCaptureOptions,
    ) => Effect.Effect<RecordingSession, AudioIoError>;
  }
>()("@speech/audio/AudioInput") {
  static readonly layer = Layer.succeed(this)({
    startRecording: Effect.fn("AudioInput.startRecording")(function* (options: AudioCaptureOptions) {
      const stream = yield* Effect.tryPromise({
        try: () =>
          navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              autoGainControl: false,
              noiseSuppression: false,
            },
          }),
        catch: toAudioIoError,
      });

      const audioContext = new AudioContext({ sampleRate: options.sampleRate });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = options.previewFftSize;
      analyser.smoothingTimeConstant = 0.15;
      source.connect(analyser);

      const recorder = new MediaRecorder(stream);
      const chunks: Array<BlobPart> = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      });
      recorder.start(250);

      const scratch = new Float32Array(analyser.fftSize);
      const previewCapacity = Math.max(
        analyser.fftSize * 4,
        Math.floor((audioContext.sampleRate * options.previewWindowMs) / 1000),
      );
      const rollingBuffer = new RollingBuffer(previewCapacity);
      const startedAt = performance.now();

      let rafId = 0;
      let active = true;
      let lastPreviewAt = 0;
      let finalized = false;
      let stopPromise: Promise<AudioSamples> | undefined;
      let sessionPeak = 0;

      const finalize = async () => {
        if (finalized) {
          return;
        }
        finalized = true;
        active = false;
        cancelAnimationFrame(rafId);
        source.disconnect();
        analyser.disconnect();
        stream.getTracks().forEach((track) => track.stop());
        await audioContext.close().catch(() => undefined);
      };

      const emitPreview = () => {
        if (!active) {
          return;
        }

        analyser.getFloatTimeDomainData(scratch);
        rollingBuffer.append(scratch);
        sessionPeak = Math.max(sessionPeak, peakAmplitude(scratch));

        const now = performance.now();
        if (now - lastPreviewAt >= options.previewIntervalMs) {
          lastPreviewAt = now;
          const frame: AudioPreviewFrame = {
            samples: rollingBuffer.snapshot(),
            sampleRate: audioContext.sampleRate,
            elapsedMs: now - startedAt,
            level: computeLevel(scratch),
            peakAmplitude: sessionPeak,
          };
          options.onFrame(frame);
        }

        rafId = requestAnimationFrame(emitPreview);
      };

      rafId = requestAnimationFrame(emitPreview);

      const stop = Effect.tryPromise({
        try: async () => {
          if (!stopPromise) {
            stopPromise = (async () => {
              const blob = await createStopPromise(recorder, chunks);
              const decoded = await decodeAudioBlob(blob, options.sampleRate);
              await finalize();
              return decoded;
            })().catch(async (error) => {
              await finalize();
              throw error;
            });
          }

          return stopPromise;
        },
        catch: toAudioIoError,
      });

      const cancel = Effect.tryPromise({
        try: async () => {
          if (recorder.state !== "inactive") {
            recorder.stop();
          }
          await finalize();
        },
        catch: toAudioIoError,
      });

      return { stop, cancel };
    }),
  });
}

export class AudioOutput extends ServiceMap.Service<
  AudioOutput,
  {
    readonly play: (audio: AudioSamples) => Effect.Effect<void, AudioIoError>;
  }
>()("@speech/audio/AudioOutput") {
  static readonly layer = Layer.succeed(this)({
    play: Effect.fn("AudioOutput.play")(function* (audio: AudioSamples) {
      yield* Effect.tryPromise({
        try: async () => {
          const context = new AudioContext({ sampleRate: audio.sampleRate });

          try {
            const buffer = context.createBuffer(1, audio.samples.length, audio.sampleRate);
            buffer.copyToChannel(new Float32Array(audio.samples), 0);
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);

            await new Promise<void>((resolve) => {
              source.addEventListener("ended", () => resolve(), { once: true });
              source.start(0);
            });
          } finally {
            await context.close().catch(() => undefined);
          }
        },
        catch: toAudioIoError,
      });
    }),
  });
}
