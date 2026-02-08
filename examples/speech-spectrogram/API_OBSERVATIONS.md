# API Observations: Using pragma-dsp in a Real App

This document records observations from building **speech-spectrogram** — a mic-recording React app that displays waveforms, spectrograms, detects pitch/formants, and lets users edit the spectrum and hear the inverse — entirely on top of `pragma-dsp`.

---

## What pragma-dsp facilitates well

### 1. `spectrum()` makes the first rung trivially easy

Getting a quick spectral snapshot of a frame required exactly one import and one function call. The scaling conventions (one-sided, amplitude-normalized) meant the resulting magnitudes were immediately interpretable — no mental math to figure out "is this in raw FFT units or dB?"

### 2. `FluentFFT` + typestate makes edit→inverse pipelines ergonomic

The spectrogram editor was the most natural part of the codebase. The code reads like the math:

```ts
const chain = fft.forward(samples).scale(s).conj();
const result = chain.inverse();
```

TypeScript caught at compile time when I accidentally tried to call `.inverse()` on a chain that lost its FFT context (e.g. from `chain()` instead of `FluentFFT.forward()`). This is genuinely useful — the "pit of success" is real.

### 3. `NonZero` branding prevents trivial mistakes

`assertNonZero(s)` before `.scale(s)` is a small ceremony, but it ensures the typestate system knows the operation is invertible. The `mulScalar(0, one)` idiom for "multiply by i" was slightly awkward but correct, and the type system kept the chain in `InverseReady` state because `im` was `NonZero`.

### 4. Split-array `ComplexArray` is friendly for visualization

Having `{ real, imag }` as separate `Float64Array`s makes it trivial to:
- compute magnitude/phase with `magnitude()` and `phase()`
- hand the arrays directly to Canvas drawing code
- reason about memory layout

No interleaved indexing gymnastics needed.

### 5. `magnitude()`, `phase()`, `binFrequencies()` remove real boilerplate

These are the functions you always end up writing. Having them as named, tested, importable helpers saved time and prevented off-by-one errors in frequency mapping.

### 6. Windowing is correct and simple

`createWindow("hann", N)` + `applyWindow(frame, window)` worked on the first try. Having `applyWindow` accept an optional `out` parameter is a nice touch for hot loops (though we didn't need it here).

---

## What pragma-dsp obstructs or doesn't cover

### 1. **No STFT / spectrogram** (the #1 gap for this use case)

We had to hand-roll a 100-line `computeStft()` that:
- loops with `hopSize` overlap
- copies frames into `Float64Array` buffers
- calls `createWindow` + `applyWindow` + `fft.forward()` + `magnitude()` per frame
- collects results into parallel arrays

This is the single most common operation in audio analysis, and every user of this library building anything visual will need it. The loop is simple but error-prone (off-by-one in hop math, forgetting to re-apply the window, etc.).

**Recommendation:** Ship `stft()` in `pragma-dsp/xform/stft` (v0.2 roadmap). Minimal API:
```ts
const result = stft(samples, { fftSize, hopSize, window, sampleRate });
// result.frames[i].magnitudes, .phases, .complex
// result.times, result.frequencies
```

### 2. **`spectrum()` discards the underlying `ComplexArray`**

`spectrum()` returns `{ frequencies, amplitude, phase, peak }` — all derived scalars. The raw complex FFT output is gone. This means:
- You **cannot** go from a `spectrum()` result back to complex data for editing or inversion
- The "edit spectrogram" workflow requires dropping down to `FFT` / `FluentFFT` directly

This creates a cliff between the beginner rung ("get a spectrum") and the power rung ("edit and invert a spectrum"). A user who starts with `spectrum()` and then wants to do anything interactive has to rewrite their pipeline.

**Recommendation:** Add an option or variant:
```ts
const result = spectrum(samples, { sampleRate, returnComplex: true });
result.complex; // ComplexArray — the raw FFT output, available for editing
```
Or a separate `spectrumComplex()` that returns both the friendly scalars and the raw data.

### 3. **`Float64Array` everywhere vs. WebAudio's `Float32Array`**

WebAudio's `AudioBuffer.getChannelData()` returns `Float32Array`. pragma-dsp's FFT, windowing, and math functions all operate on and return `Float64Array`. This requires conversion at every boundary:
- Decoding: `Float32Array` → copy into `Float64Array` for FFT input
- Playback: `Float64Array` → copy into `Float32Array` for `AudioContext`
- Rendering: `Float64Array` from magnitude → fine for Canvas (it coerces), but wasteful

The conversions aren't hard, but they add noise and allocations.

**Recommendation:** Consider accepting `Float32Array` in FFT `.forward()` (it already accepts `ArrayLike<number>`, so `Float32Array` works as input!). The real friction is that _output_ is always `Float64Array`. For a future audio-focused layer, `Float32Array` output variants could help.

*Note:* On reflection, `forward()` accepting `ArrayLike<number>` means `Float32Array` input already works. The friction is only on the output side, and `Float64Array` is the right default for precision. This is mostly a documentation gap rather than an API gap.

### 4. **No autocorrelation helper**

Pitch detection via autocorrelation is `FFT → |·|² → IFFT` — three lines of pragma-dsp calls — but it's a well-known pattern that users shouldn't have to rediscover. We had to:
- Manually create the power spectrum `ComplexArray`
- Know to set the imaginary part to 0
- Remember to zero-pad for linear (not circular) autocorrelation

**Recommendation:** Add `autocorrelation(samples)` to `pragma-dsp/xform/fourier`:
```ts
const r = autocorrelation(samples); // Float64Array, r[lag]
```
It's a thin wrapper but eliminates a sharp edge.

### 5. **No spectral envelope / smoothing utilities**

Formant detection required a hand-rolled moving-average smoother and peak-picker. These are generic enough to be reusable.

**Recommendation (lower priority):** This might belong in a future `pragma-dsp/analysis` module rather than in the core Fourier API.

### 6. **FluentFFT allocates a new FFT instance every call**

In the spectrogram editor, each edit preset creates `new FluentFFT(fftSize)`. Each instance creates a new `Radix2Fft` with twiddle tables and bit-reversal permutation. For the editor this is fine (runs once per user interaction), but in a hot loop (STFT frames) it would be wasteful.

The Effect layer (`FourierLive`) solves this via caching, but non-Effect users need to manually cache `FluentFFT` instances. This is an expected trade-off but worth documenting.

### 7. **Clone-before-inverse ceremony in the editor**

Because fluent operations mutate in-place, the editor pattern is:
```ts
const chain = fft.forward(samples).scale(s);
const editedComplex = chain.clone().unwrap(); // need this for spectrogram display
const result = chain.inverse();               // mutates chain's data
```

You must `.clone()` before `.inverse()` if you want to keep the edited frequency-domain data for display. This is documented behavior, but in practice it's a common footgun. 

**Recommendation:** Consider an `into(out)` variant on `.inverse()` that writes into a separate buffer without consuming the chain's data. Alternatively, document the clone-before-inverse pattern prominently.

---

## Summary of proposed API changes

| Priority | Change | Rationale |
|----------|--------|-----------|
| **High** | `stft()` in `pragma-dsp/xform/stft` | #1 gap for any audio/visualization use case |
| **High** | `spectrum()` option to return raw `ComplexArray` | Bridges beginner→power rung without rewrite |
| **Medium** | `autocorrelation()` in `pragma-dsp/xform/fourier` | Common pattern, easy to implement wrong |
| **Medium** | Document that `Float32Array` input already works in `forward()` | Users assume they need Float64 |
| **Low** | `inverse(out)` or non-consuming inverse on `ComplexChain` | Avoids clone-before-inverse pattern |
| **Low** | Spectral smoothing / envelope helper | Useful for formant detection, but niche |

---

## Overall assessment

pragma-dsp's "API ladder" design genuinely works. The beginner rung (`spectrum()`) got us a quick result; the power rung (`FFT` + helpers) gave us full control for STFT; the fluent rung (`FluentFFT`) made the spectrogram editor feel like writing math. The main gap is the missing middle ground between "one frame" and "many overlapping frames" — i.e., STFT — which is already on the v0.2 roadmap. The library is a good foundation; the friction points are addressable without breaking changes.
