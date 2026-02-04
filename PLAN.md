# pragma-dsp v0.1 Plan

## Why pragma-dsp exists

pragma-dsp is a **pragmatic** DSP library for TypeScript. The goal is not to win microbenchmarks; it is to be **useful**:

- Easy on-ramp for beginners who just want a spectrum and peak frequency
- Flexible, low-friction building blocks for experts
- Portable across runtimes by staying **pure typed-array math** with no environment coupling
- Reliable, testable implementations that agents can extend safely via strong test and fixture harnesses

Tagline we keep in mind: **It is not meant to be fast, it is meant to be useful.**
“Useful” still implies “reasonably fast” and “does not allocate unnecessarily,” but correctness and DX come first.

## Core design philosophy: APIs as ladders

We follow “APIs as ladders”:
https://blog.sbensu.com/posts/apis-as-ladders/

A good ladder has:
- A first rung that feels effortless and obvious
- A second rung that exposes common power knobs
- Lower rungs that reveal raw capabilities without forcing complexity on everyone

In pragma-dsp, that maps to:

1. **Beginner rung** (`pragma-dsp`): `spectrum(samples, options?)`
2. **Power rung** (`pragma-dsp/xform/fourier`): `FFT`, windowing, magnitude/phase, frequency mapping
3. **Expert rung** (`pragma-dsp/core`): low-level primitives, buffer reuse, minimal allocations

Effect integration is an optional parallel ladder:
- **Effect rung** (`pragma-dsp/effect`): services/layers/streams for async and realtime pipelines

## v0.1 scope (what we ship and why)

### Ship in v0.1
- Fourier transform (radix-2 FFT + IFFT)
- Window functions (small set, high utility): rect, hann, hamming, blackman
- Helpers that remove common friction:
  - magnitude and phase
  - fftShift
  - frequency axis mapping (binFrequencies)
  - peak detection
- Beginner-first `spectrum()` that returns frequencies, amplitude, phase, peak

### Do not ship in v0.1
- STFT, sliding DFT, DCT/MDCT, wavelets, NTT, filters, time-frequency distributions
- Any platform I/O (audio decoding, microphone capture, filesystem utilities)

Reason: v0.1 is about establishing:
- a reliable FFT kernel
- a stable API ladder shape
- a robust correctness + fixture + benchmark harness for agents

Everything else builds on that foundation.

## Module boundaries (API structure + project structure)

We optimize for:
- **Short imports** (avoid deep namespacing like `Transform.Fourier.*`)
- **Tree shaking** (users pay only for what they import)
- **Future growth** (new transforms do not bloat beginner entrypoint)

Public entrypoints:
- `pragma-dsp` (beginner)
- `pragma-dsp/xform/fourier` (power Fourier)
- `pragma-dsp/core` (expert low-level)
- `pragma-dsp/effect` (optional Effect wrappers)

Recommended import examples:
```ts
import { spectrum } from "pragma-dsp";

import {
  FFT,
  createWindow,
  magnitude,
  phase,
  fftShift,
  binFrequencies
} from "pragma-dsp/xform/fourier";

import { FourierLive, spectrumStream } from "pragma-dsp/effect";

Project layout (source):
- `src/index.ts` exports only beginner API
- `src/public/*` implements beginner API
- `src/xform/fourier.ts` exports Fourier “power rung”
- `src/core/*` contains internal building blocks (no side effects)
- `src/effect/*` contains Effect Tag/Layer/Stream helpers

## Runtime portability rules

The core must be “engine-only” JavaScript:
- Allowed: `Math`, `TypedArray`, pure functions, deterministic classes
- Not allowed in core: `fs`, `path`, `Buffer`, `process`, `window`, `document`, `AudioContext`, `fetch`

Reason: a useful DSP library should run unchanged in:
- Node, Bun
- Browsers (bundled)
- Edge/workers (bundled)
- React Native (Metro bundling)

## Fourier conventions (v0.1)

### FFT sizes
- FFT implementation supports radix-2 sizes only in v0.1.
- Beginner APIs default to `nextPow2(signal.length)`.

### Output representations
- Power rung uses split arrays: `{ real, imag }`.
- Beginner rung returns amplitude and phase directly.

### Amplitude scaling and “useful defaults”
For beginner `spectrum()` we prioritize interpretability:
- `sides: "one"` (default): return `N/2 + 1` bins `[0..Nyquist]`
  - amplitude scaling so a bin-centered sine of amplitude `A` has peak amplitude approximately `A`:
    - `amp[0] = |X[0]| / N` (DC, not doubled)
    - `amp[N/2] = |X[N/2]| / N` when `N` even (Nyquist, not doubled)
    - other bins: `amp[k] = (2 * |X[k]|) / N`
- `sides: "two"`: return `N` bins, no doubling:
  - `amp[k] = |X[k]| / N`

Phase:
- `phase[k] = atan2(im[k], re[k])`

Peak detection:
- default ignores DC bin unless DC is the only meaningful component

## Effect integration (v0.1)

Effect is optional and should not impact non-Effect users.
v0.1 provides:
- `Fourier` service Tag that caches FFT instances and windows
- `FourierLive` layer using in-memory Maps
- `spectrumFx` wrapper around `spectrum()`
- `spectrumStream(frames, opts)` mapping `Stream<Float32Array>` to spectra

Reason: streaming and async pipelines (websocket, mic frames, sensors, time-series) benefit from:
- structured dependency injection
- caching with explicit lifetimes
- composable streaming + backpressure + cancellation

## Correctness and fixtures: agent-first TDD

We will use agent-first TDD:
- Agents are good at writing exhaustive tests and iterating until green.
- Our harness should prevent subtle regressions and scaling/sign mistakes.

### Testing strategy
1. Naive DFT reference for small N (8,16,32) to validate FFT forward output.
2. Round-trip property: `ifft(fft(x)) ≈ x`
3. Known-signal tests: bin-centered sine amplitude and peak frequency
4. Window sanity tests (Hann endpoints near 0, symmetry)

### Golden fixtures generated by Python
To reduce “self-consistency” traps, we generate fixtures using trusted libraries:
- NumPy for FFT reference
- SciPy for windows reference (when available)

Fixtures live in:
- `test/fixtures/*.json`

A generator script lives in:
- `scripts/gen_fixtures.py`

Reason: correctness must not depend solely on our own implementation.

## Benchmarks: guardrails, not bragging

Benchmarks exist to:
- ensure we’re not accidentally slow
- detect big regressions
- keep allocation patterns reasonable

Benchmark design:
- Benchmarks consume the same JSON fixtures used for correctness.
- Each benchmark run:
  - loads fixture inputs
  - runs FFT forward on representative sizes (e.g. 2048, 4096)
  - computes a cheap checksum of outputs to prevent dead-code elimination
  - optionally validates the checksum against fixture-derived expected values once per run (not per iteration)

Reason: bench inputs should be stable across machines and agent edits.

## Tree-shaking requirements

- `package.json` uses `exports` subpaths
- `"sideEffects": false`
- avoid default-exported mega objects
- avoid import-time side effects and caches
- prefer named exports

## v0.1 implementation checklist (what agents should do)

1. Implement core FFT (radix-2 iterative Cooley–Tukey) with twiddle tables and bit-reversal.
2. Implement Fourier power API (`FFT`, windowing, magnitude/phase, fftShift, binFrequencies).
3. Implement beginner `spectrum()` with scaling rules and peak detection.
4. Implement Python fixture generator and commit fixtures.
5. Implement tests:
   - DFT comparison
   - round-trip
   - sine scaling
   - window sanity
6. Implement benchmarks reading JSON fixtures and reporting timings.
7. Implement Effect layer:
   - Fourier Tag, FourierLive Layer, spectrumFx, spectrumStream
8. Ensure build outputs match `exports` entrypoints and `pragma-dsp` import does not require `effect`.
