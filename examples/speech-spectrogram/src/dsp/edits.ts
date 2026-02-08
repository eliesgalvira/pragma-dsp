/**
 * Spectrogram editing using pragma-dsp's FluentFFT + typestate chain.
 *
 * Each "edit" is a named transform that takes a ComplexChain from
 * FluentFFT.forward() and applies operations, then .inverse() to
 * reconstruct the time-domain signal.
 */
import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";
import { assertNonZero, type NonZero } from "pragma-dsp/fluent";
import { createComplexArray, type ComplexArray } from "pragma-dsp/core";

export type EditKind =
  | { type: "scale"; factor: number }
  | { type: "multiply_by_i" }
  | { type: "conjugate" }
  | { type: "negate" }
  | { type: "scale_and_conjugate"; factor: number };

export const EDIT_PRESETS: { label: string; edit: EditKind }[] = [
  { label: "Scale ×4", edit: { type: "scale", factor: 4 } },
  { label: "Scale ×0.25", edit: { type: "scale", factor: 0.25 } },
  { label: "Multiply by i", edit: { type: "multiply_by_i" } },
  { label: "Conjugate", edit: { type: "conjugate" } },
  { label: "Negate", edit: { type: "negate" } },
  { label: "Scale ×2 + Conjugate", edit: { type: "scale_and_conjugate", factor: 2 } },
];

/**
 * Apply a spectral edit to a time-domain frame and return the
 * reconstructed (inverse FFT) signal.
 *
 * This exercises the FluentFFT → chain → inverse() pipeline.
 */
export function applySpectralEdit(
  samples: Float64Array,
  fftSize: number,
  edit: EditKind
): { edited: Float64Array; editedComplex: ComplexArray } {
  const fft = new FluentFFT(fftSize);
  const out = createComplexArray(fftSize);

  switch (edit.type) {
    case "scale": {
      const s = edit.factor;
      assertNonZero(s);
      const chain = fft.forward(samples).scale(s);
      const editedComplex = chain.unwrap();
      const result = chain.inverseInto(out);
      return { edited: result.real, editedComplex };
    }

    case "multiply_by_i": {
      // Multiply every bin by i = (0 + 1i)
      // mulScalar(0, 1) — 1 is NonZero so im param preserves invertibility
      const one = 1 as NonZero; // 1 is trivially nonzero
      const chain = fft.forward(samples).mulScalar(0, one);
      const editedComplex = chain.unwrap();
      const result = chain.inverseInto(out);
      return { edited: result.real, editedComplex };
    }

    case "conjugate": {
      const chain = fft.forward(samples).conj();
      const editedComplex = chain.unwrap();
      const result = chain.inverseInto(out);
      return { edited: result.real, editedComplex };
    }

    case "negate": {
      const s = -1;
      assertNonZero(s);
      const chain = fft.forward(samples).scale(s);
      const editedComplex = chain.unwrap();
      const result = chain.inverseInto(out);
      return { edited: result.real, editedComplex };
    }

    case "scale_and_conjugate": {
      const s = edit.factor;
      assertNonZero(s);
      const chain = fft.forward(samples).scale(s).conj();
      const editedComplex = chain.unwrap();
      const result = chain.inverseInto(out);
      return { edited: result.real, editedComplex };
    }
  }
}
