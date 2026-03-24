import { createComplexArray } from "pragma-dsp/core";
import { FluentFFT } from "pragma-dsp/xform/fourier-fluent";

import type { SpectralEditKind } from "./domain";

const nextPowerOfTwo = (value: number) => {
  let power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
};

export const applySpectralEdit = (
  samples: Float32Array,
  edit: SpectralEditKind,
) => {
  const fftSize = nextPowerOfTwo(samples.length);
  const fft = new FluentFFT(fftSize);
  const input = new Float64Array(fftSize);
  input.set(samples);
  const output = createComplexArray(fftSize);

  switch (edit.type) {
    case "identity":
      return input;

    case "complex_multiply": {
      if (edit.real === 1 && edit.imaginary === 0) {
        return input;
      }

      const chain = fft.forward(input).mulScalar(edit.real, edit.imaginary);
      return chain.inverseInto(output).real;
    }
  }
};
