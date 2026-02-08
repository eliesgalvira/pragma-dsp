/**
 * Fluent FFT entrypoint — opt-in subpath export.
 *
 * Provides `FluentFFT`, a thin wrapper around `FFT` whose `.forward()` returns
 * a `ComplexChain` with inverse context bound, enabling fluent pipelines like:
 *
 * ```ts
 * import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";
 * import { assertNonZero } from "pragma-dsp/fluent";
 *
 * const fft = new FluentFFT(1024);
 * const s = 2; assertNonZero(s);
 *
 * const result = fft
 *   .forward(signal)
 *   .scale(s)
 *   .conj()
 *   .inverse();
 * ```
 *
 * @module
 */

import type { ComplexArray } from "../core/fft.js";
import { FFT } from "../xform/fourier.js";
import { ComplexChain, type FftForwardState } from "../fluent/complex.js";

/**
 * A fluent-friendly FFT wrapper.
 *
 * Same radix-2 kernel as `FFT`, but `.forward()` returns a `ComplexChain`
 * with `hasFft: true` so that `.inverse()` is available at the end of a
 * fluent chain.
 */
export class FluentFFT {
  readonly size: number;
  private readonly fft: FFT;

  constructor(size: number) {
    this.fft = new FFT(size);
    this.size = this.fft.size;
  }

  /**
   * Forward FFT, returning a fluent `ComplexChain` with inverse context bound.
   *
   * The returned chain is in typestate `FftForwardState`:
   *   `{ kind:"complex", hasFft:true, invert:"yes", len:"same" }`
   *
   * so `.inverse()` is directly callable after invertible operations.
   */
  forward(input: ArrayLike<number>, out?: ComplexArray): ComplexChain<FftForwardState> {
    const data = this.fft.forward(input, out);
    return new ComplexChain<FftForwardState>(
      data,
      (d, o) => this.fft.inverse(d, o),
    );
  }

  /**
   * Forward FFT for complex input.
   */
  forwardComplex(input: ComplexArray, out?: ComplexArray): ComplexChain<FftForwardState> {
    const data = this.fft.forwardComplex(input, out);
    return new ComplexChain<FftForwardState>(
      data,
      (d, o) => this.fft.inverse(d, o),
    );
  }
}
