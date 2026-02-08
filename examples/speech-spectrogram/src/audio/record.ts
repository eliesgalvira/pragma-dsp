import { Deferred } from "./deferred";

// ── Mic access ───────────────────────────────────────────────────────
export function getUserMicrophoneStream(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      autoGainControl: false,
      noiseSuppression: false,
    },
  });
}

// ── Record → Blob (deferred pattern) ────────────────────────────────
/**
 * Start recording from `stream`.
 * Returns a *stop* function – calling it stops the recorder and
 * returns a Promise<Blob> that resolves once the MediaRecorder has
 * flushed all data (via the deferred pattern).
 */
export function recordStreamAsBlob(stream: MediaStream) {
  const mediaRecorder = new MediaRecorder(stream);

  const chunks: BlobPart[] = [];
  const deferredBlob = new Deferred<Blob, Error>();

  mediaRecorder.addEventListener("dataavailable", (e) => {
    chunks.push(e.data);
  });

  mediaRecorder.addEventListener("stop", () => {
    const blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
    deferredBlob.resolve(blob);
  });

  mediaRecorder.start();

  return () => {
    mediaRecorder.stop();
    return deferredBlob.promise;
  };
}

// ── Blob → Float32Array decode ──────────────────────────────────────
/**
 * Decode an audio Blob into a mono Float32Array at `sampleRate`.
 * Uses OfflineAudioContext for reliable cross-browser decoding.
 */
export async function blobToFloat32Array(
  blob: Blob,
  sampleRate: number = 16_000
): Promise<{ samples: Float32Array; sampleRate: number }> {
  const arrayBuffer = await blob.arrayBuffer();

  // Create a long-enough offline context – 60 s should cover any recording
  const ctx = new OfflineAudioContext(1, sampleRate * 60, sampleRate);
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

  // Grab first channel (mono)
  const raw = audioBuffer.getChannelData(0);

  // Trim to actual length (audioBuffer.length ≤ 60s * sampleRate)
  const samples = new Float32Array(audioBuffer.length);
  samples.set(raw.subarray(0, audioBuffer.length));

  return { samples, sampleRate: audioBuffer.sampleRate };
}

// ── Playback helper ─────────────────────────────────────────────────
/**
 * Play a Float32Array signal through the default audio output.
 * Returns a promise that resolves when playback finishes.
 */
export function playSignal(
  samples: Float32Array | Float64Array,
  sampleRate: number
): Promise<void> {
  const ctx = new AudioContext({ sampleRate });
  const buffer = ctx.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);

  // Copy — handles both Float32Array and Float64Array
  for (let i = 0; i < samples.length; i++) {
    channel[i] = samples[i]!;
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  return new Promise<void>((resolve) => {
    source.addEventListener("ended", () => {
      ctx.close();
      resolve();
    });
    source.start();
  });
}
