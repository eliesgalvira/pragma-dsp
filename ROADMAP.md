# pragma-dsp Roadmap

This roadmap explains how pragma-dsp will grow beyond v0.1 while preserving:
- usefulness (sane defaults, good ergonomics)
- the API ladder (beginner -> power -> expert)
- tree-shaking and modular imports
- runtime portability

## Guiding principles for new modules

1. Beginner API stays small
- The `pragma-dsp` root entrypoint should remain a curated, high-leverage set of functions.
- New transforms generally land in `pragma-dsp/xform/*` first.
- Beginner entrypoint may expose a new one-liner only if it is extremely common and stable.

2. Subpath exports are the primary UX tool
- Prefer `import { STFT } from "pragma-dsp/xform/stft"` over nested namespaces.
- Avoid designs that force `xform.Fourier.STFT.*`.

3. Each transform family gets:
- a power rung module in `xform/*`
- low-level pieces in `core/*` if needed
- optional Effect services/streams in `effect/*` if streaming or caching is meaningful

4. Defaults should be “pragmatic”
- Provide safe defaults for common tasks (audio analysis, visualization).
- Document scaling conventions and how to override them.

## Planned modules

### A) STFT (Short-Time Fourier Transform)
Why: STFT is the mainstream tool for audio analysis and spectrograms.

Proposed modules:
- `pragma-dsp/xform/stft`
  - `stft(signal, opts)` returning frames x frequency bins (magnitude/phase/power)
  - `spectrogram(signal, opts)` convenience wrapper returning a flattened heatmap format
- `pragma-dsp/effect/stft`
  - streaming STFT over `Stream<Float32Array>` frames
  - backpressure-friendly spectrogram pipeline

Key options:
- `fftSize`, `hopSize`, `window`, `sampleRate`
- output: magnitude, power, complex
- scaling modes appropriate for audio visualization

### B) Sliding DFT
Why: real-time frequency tracking with incremental updates; useful in monitoring and interactive systems.

Proposed modules:
- `pragma-dsp/xform/sliding-dft`
  - `SlidingDFT` class that updates spectrum each sample/hop
- `pragma-dsp/effect/sliding-dft`
  - stream adapter for sample streams or small frame streams

Notes:
- This is an “expert” tool; likely not exposed in root beginner API.

### C) DCT (Discrete Cosine Transform)
Why: core tool for compression and many real-only transforms; useful and relatively approachable.

Proposed module:
- `pragma-dsp/xform/dct`
  - `dct(signal, { type })`, `idct(...)`
  - consider DCT-II and DCT-III first

Can be implemented using FFT internally where appropriate.

### D) MDCT (Modified Discrete Cosine Transform)
Why: modern lossy audio coding building block (overlap-add).

Proposed module:
- `pragma-dsp/xform/mdct`
  - `mdct(signal, opts)`, `imdct(...)`
  - overlap-add utilities

Notes:
- More domain-specific than DCT; keep it modular and well-documented.

### E) Wavelets (DWT)
Why: time-localized analysis; useful for denoising, multi-resolution analysis, and education.

Proposed module:
- `pragma-dsp/xform/wavelet`
  - start with Haar + a small Daubechies set
  - forward/inverse DWT, multi-level decomposition

### F) NTT / finite-field transforms
Why: polynomial arithmetic for ECC, ZK, and cryptography; niche but valuable if implemented well in TS.

Proposed module:
- `pragma-dsp/xform/ntt` (or `pragma-dsp/ff` if you want a clearer finite-field namespace)
  - field abstractions (prime fields first)
  - NTT forward/inverse
  - polynomial multiplication utilities

Notes:
- Keep this separated from numeric FFT to avoid complexity and bundle size for DSP users.

## Filters and utilities

### Filters
Proposed:
- `pragma-dsp/filters`
  - FIR helpers, convolution
  - windowed-sinc filter design helpers
  - frequency response analysis via FFT utilities

### Time series utilities (optional)
Proposed:
- `pragma-dsp/ts`
  - resampling to uniform grid
  - detrending, normalization, smoothing
This supports domains like trading and sensor analysis without coupling to any platform.

## Effect integration growth

Effect modules should remain optional and provide:
- caching (FFT instances, windows, twiddles, kernels)
- streaming adapters (STFT over streams)
- typed error boundaries
- test-friendly service substitution

Proposed additions:
- `pragma-dsp/effect/audio` (maybe later, and likely in a separate package)
  - integrations with WebAudio pipelines belong outside the core library to preserve portability

## Versioning guidance

- v0.1: FFT + windows + basic spectrum + Effect caching/stream mapping
- v0.2: STFT + spectrogram convenience + more windows
- v0.3: DCT, maybe Hilbert/analytic signal helpers
- v0.4+: sliding DFT, MDCT
- v1.0: after API conventions (scaling, naming, module layout) have stabilized in real usage

We do not promise these versions; the ordering reflects pragmatic utility and dependency depth.
