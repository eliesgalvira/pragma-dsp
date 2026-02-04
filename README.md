# pragma-dsp

Useful DSP primitives for TypeScript with a beginner-friendly ladder of APIs.

## Use-case examples

### 1) Beginner spectrum (one-liner)
```ts
import { spectrum } from "pragma-dsp";

const samples = Float32Array.from([0, 1, 0, -1, 0, 1, 0, -1]);
const result = spectrum(samples, { sampleRate: 48_000 });

console.log(result.peak.frequency, result.peak.amplitude);
```

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

### 3) Expert core reuse (manual buffers)
```ts
import { Radix2Fft, createComplexArray } from "pragma-dsp/core";

const fft = new Radix2Fft(1024);
const out = createComplexArray(1024);
const input = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i));

fft.forward(input, out);
// Reuse `out` across frames to avoid allocations.
```

### 4) Effect integration (optional)
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
