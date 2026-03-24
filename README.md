# pragma-dsp

Useful DSP primitives for TypeScript with a beginner-friendly ladder of APIs.

## Use-case examples

### 1) Beginner spectrum (one-liner)
```ts
import { spectrum } from "pragma-dsp";

const samples = Float32Array.from([0, 1, 0, -1, 0, 1, 0, -1]);
const result = spectrum(samples, { sampleRate: 48_000, returnComplex: true });

console.log(result.peak.frequency, result.peak.amplitude);
console.log(result.complex.real.length); // raw FFT bins
```

Note: inputs may be `Float32Array` (e.g. WebAudio), while outputs are `Float64Array` for precision.

### 2) Power Fourier utilities
```ts
import {
  FFT,
  createWindow,
  magnitude,
  phase,
  binFrequencies
} from "pragma-dsp/xform/fourier";

const fftSize = 1024;
const fft = new FFT(fftSize);
const window = createWindow("hann", fftSize);
const input = Float32Array.from({ length: fftSize }, (_, i) => Math.sin(i));

const windowed = input.map((v, i) => v * window[i]);
const complex = fft.forward(windowed);
const mag = magnitude(complex);
const ang = phase(complex);
const freqs = binFrequencies(fftSize, 48_000, "one");

console.log(freqs[10], mag[10], ang[10]);
```

### 3) Short-time Fourier transform (STFT)
```ts
import { stft } from "pragma-dsp/xform/stft";

const samples = Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i));
const result = stft(samples, {
  fftSize: 1024,
  hopSize: 256,
  sampleRate: 48_000,
  window: "hann"
});

console.log(result.frames.length, result.frequencies.length);
```

### 4) Spectrogram rung
```ts
import { spectrogram } from "pragma-dsp/xform/stft";

const result = spectrogram(samples, {
  sampleRate: 48_000,
  fftSize: 1024,
  hopSize: 256,
  floorDb: -90
});

console.log(result.dbFrames.length, result.stft.times.length);
```

### 5) Speech-analysis rung
```ts
import { analyzeSpeech } from "pragma-dsp/analysis";

const analysis = analyzeSpeech(samples, {
  sampleRate: 16_000,
  fftSize: 1024,
  hopSize: 256,
});

console.log(analysis.medianF0, analysis.formantMedians);
```

### 6) Expert core reuse (manual buffers)
```ts
import { Radix2Fft, createComplexArray } from "pragma-dsp/core";

const fft = new Radix2Fft(1024);
const out = createComplexArray(1024);
const input = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i));

fft.forward(input, out);
// Reuse `out` across frames to avoid allocations.
```

### 7) Effect integration (optional)
```ts
import { Stream } from "effect";
import { FourierLive, spectrumStream } from "pragma-dsp/effect";

const frames = Stream.fromIterable([
  Float32Array.from([0, 1, 0, -1]),
  Float32Array.from([1, 0, -1, 0])
]);

const spectra = spectrumStream(frames, { sampleRate: 48_000 });
// Provide FourierLive in your Layer stack to enable caching.
```
