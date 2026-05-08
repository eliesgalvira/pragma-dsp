# pragma-dsp

Pragmatic DSP primitives for TypeScript.

`pragma-dsp` is designed as an API ladder:

- **Beginner**: one-liners for common tasks like “what is the dominant frequency?”
- **Power**: STFT, spectrogram, speech-analysis helpers, and reusable FFT utilities
- **Expert**: raw kernels, reusable buffers, and opt-in fluent spectral pipelines

The goal is not to hide DSP. The goal is to let users start at the simplest rung that solves their problem, then move downward only when they need more control.

## Installation

```bash
npm install pragma-dsp@alpha
```

```bash
pnpm add pragma-dsp@alpha
```

```bash
bun add pragma-dsp@alpha
```

If you want the optional Effect integration:

```bash
npm install pragma-dsp@alpha effect
```

```bash
pnpm add pragma-dsp@alpha effect
```

```bash
bun add pragma-dsp@alpha effect
```

## What This Library Is Good For

- Audio spectrum inspection
- Spectrogram generation
- Speech-oriented analysis helpers
- FFT-based transforms and experiments
- Reusable DSP building blocks for apps, demos, and libraries

## API Ladder

| Rung | Import | Use when |
|---|---|---|
| Beginner | `pragma-dsp` | You want a useful answer fast |
| Power | `pragma-dsp/xform/*`, `pragma-dsp/analysis` | You need explicit transform control or analysis workflows |
| Fluent | `pragma-dsp/xform/fourier-fluent`, `pragma-dsp/fluent` | You want readable spectral-edit pipelines with inverse support |
| Expert | `pragma-dsp/core`, `pragma-dsp/math/complex` | You need buffer reuse, low-level control, or custom orchestration |
| Optional Effect | `pragma-dsp/effect` | You want cached FFT/window services or stream integration |

## Quick Start

### 1) Beginner spectrum

```ts
import { spectrum } from "pragma-dsp";

const samples = Float32Array.from([0, 1, 0, -1, 0, 1, 0, -1]);

const result = spectrum(samples, {
  sampleRate: 48_000,
  returnComplex: true
});

console.log(result.peak.frequency);
console.log(result.peak.amplitude);
console.log(result.complex.real.length);
```

This is the “just give me a useful spectrum” API.

It:
- zero-pads to the next power of two by default
- applies a window
- computes frequencies, amplitude, and phase
- detects the dominant peak

## Power APIs

### Fourier utilities

Use this rung when you want manual control over FFT size, windows, magnitudes, phases, and bin frequencies.

```ts
import {
  FFT,
  createWindow,
  applyWindow,
  magnitude,
  phase,
  binFrequencies
} from "pragma-dsp/xform/fourier";

const fftSize = 1024;
const fft = new FFT(fftSize);
const window = createWindow("hann", fftSize);
const input = Float32Array.from({ length: fftSize }, (_, i) => Math.sin(i));

const windowed = applyWindow(input, window);
const complex = fft.forward(windowed);
const mag = magnitude(complex);
const ang = phase(complex);
const freqs = binFrequencies(fftSize, 48_000, "one");

console.log(freqs[10], mag[10], ang[10]);
```

### STFT

Use this when a single FFT is not enough and you need time-varying frequency content.

```ts
import { stft } from "pragma-dsp/xform/stft";

const samples = Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i));

const result = stft(samples, {
  fftSize: 1024,
  hopSize: 256,
  sampleRate: 48_000,
  window: "hann"
});

console.log(result.frames.length);
console.log(result.times.length);
console.log(result.frequencies.length);
```

### Spectrogram

Use this when you want display-ready STFT magnitudes in dB instead of rebuilding that conversion in app code.

```ts
import { spectrogram } from "pragma-dsp/xform/stft";

const samples = Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i));

const result = spectrogram(samples, {
  sampleRate: 48_000,
  fftSize: 1024,
  hopSize: 256,
  floorDb: -90
});

console.log(result.dbFrames.length);
console.log(result.stft.times.length);
console.log(result.stft.frequencies.length);
```

### Speech analysis

Use this rung when you want a speech-oriented workflow rather than stitching together STFT, pitch, and formant helpers manually.

```ts
import { analyzeSpeech } from "pragma-dsp/analysis";

const samples = Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i));

const analysis = analyzeSpeech(samples, {
  sampleRate: 16_000,
  fftSize: 1024,
  hopSize: 256
});

console.log(analysis.medianF0);
console.log(analysis.formantMedians);
```

Also available from `pragma-dsp/analysis`:

- `detectPitch()`
- `detectFormants()`
- `magnitudeToDb()`
- `movingAverage()`
- `spectralEnvelope()`

## Fluent API

The fluent API is for readable spectral-edit pipelines.

It exists because there is a gap between:
- “give me a spectrum”
- “manually mutate split complex arrays and manage inverse transforms myself”

### What it does

- wraps complex FFT output in a chainable object
- keeps inverse FFT context when the chain came from `FluentFFT.forward()`
- lets TypeScript gate `.inverse()` to chains that are still statically safe to invert

### Important behavior

- **Fluent operations mutate in place**
- call `.clone()` if you want persistence before mutating
- use `.inverseChecked()` when invertibility depends on runtime values

### Minimal fluent example

```ts
import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";
import { assertNonZero } from "pragma-dsp/fluent";

const fft = new FluentFFT(1024);
const signal = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i));
const gain = 2;
assertNonZero(gain);

const edited = fft
  .forward(signal)
  .scale(gain)
  .conj()
  .inverse();
```

### Runtime-checked fluent example

```ts
import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";
import { createComplexArray } from "pragma-dsp/core";

const fft = new FluentFFT(1024);
const signal = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i));
const mask = createComplexArray(1024);

const result = fft
  .forward(signal)
  .mul(mask)
  .inverseChecked();

if (!result.ok) {
  console.error(result.error);
} else {
  console.log(result.value.real[0]);
}
```

### When to use Fluent instead of plain FFT utilities

Use Fluent when:
- you are doing **spectral edits**
- you want code that reads like the transform you are applying
- you want inverse availability encoded in the chain

Use plain `FFT` utilities when:
- you want explicit arrays and explicit steps
- you are inspecting magnitudes/phases rather than editing
- you want the smallest possible abstraction

## Expert APIs

Use `pragma-dsp/core` when you want raw reusable pieces with minimal abstraction overhead.

```ts
import { Radix2Fft, createComplexArray } from "pragma-dsp/core";

const fft = new Radix2Fft(1024);
const out = createComplexArray(1024);
const input = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i));

fft.forward(input, out);
```

Use `pragma-dsp/math/complex` when you want pure complex-array arithmetic without the fluent wrapper.

## Effect Integration

The Effect module is optional.

Use it when you want:
- cached FFT instances
- cached windows
- stream-based spectrum computation

```ts
import { Stream } from "effect";
import { FourierLive, spectrumStream } from "pragma-dsp/effect";

const frames = Stream.fromIterable([
  Float32Array.from([0, 1, 0, -1]),
  Float32Array.from([1, 0, -1, 0])
]);

const spectra = spectrumStream(frames, { sampleRate: 48_000 });

// Provide FourierLive in your Layer stack.
void spectra;
void FourierLive;
```

## Module Layout

### `pragma-dsp`

Curated beginner API.

Currently:
- `spectrum()`

### `pragma-dsp/xform/fourier`

Power Fourier utilities.

Examples:
- `FFT`
- `applyWindow`
- `createWindow`
- `magnitude`
- `phase`
- `binFrequencies`

### `pragma-dsp/xform/stft`

Short-time Fourier transforms and spectrogram helpers.

Examples:
- `stft()`
- `spectrogram()`

### `pragma-dsp/analysis`

Higher-level analysis helpers for audio and speech workflows.

Examples:
- `analyzeSpeech()`
- `detectPitch()`
- `detectFormants()`
- `magnitudeToDb()`
- `movingAverage()`
- `spectralEnvelope()`

### `pragma-dsp/xform/fourier-fluent`

Fluent FFT entrypoint.

Examples:
- `FluentFFT`

### `pragma-dsp/fluent`

Fluent helpers and chain types.

Examples:
- `ComplexChain`
- `chain()`
- `assertNonZero()`
- `asNonZero()`

### `pragma-dsp/core`

Low-level kernels and reusable storage helpers.

### `pragma-dsp/math/complex`

Pure complex-array math, tree-shakeable and independent of FFT orchestration.

### `pragma-dsp/effect`

Optional Effect wrappers and stream integration.

## Design Notes

### Typed arrays in, typed arrays out

This library is built around typed arrays and split complex arrays:

```ts
type ComplexArray = {
  real: Float64Array;
  imag: Float64Array;
};
```

Inputs may be `Float32Array` or other `ArrayLike<number>`. Outputs are generally `Float64Array` for numerical stability and predictable downstream math.

### Tree-shaking

The library is intentionally split into subpath exports so users do not pay for higher rungs they do not import.

### Optional Effect dependency

`effect` is a peer dependency for the `pragma-dsp/effect` entrypoint only. The rest of the library is dependency-free.

## Current Status

Shipped:
- `spectrum()`
- Fourier utilities
- STFT
- spectrogram helper
- speech-analysis helper rung
- fluent FFT pipeline
- optional Effect wrappers

Planned next steps live in [ROADMAP.md](./ROADMAP.md).

## Development

```bash
pnpm install
pnpm exec tsc --noEmit
pnpm exec vitest run test
```

Useful scripts:

```bash
pnpm run build
pnpm run bench
pnpm run gen:fixtures
pnpm run gen:refs
```
