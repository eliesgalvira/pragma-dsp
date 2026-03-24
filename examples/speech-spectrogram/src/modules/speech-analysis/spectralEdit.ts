import { createComplexArray } from "pragma-dsp/core";
import { assertNonZero } from "pragma-dsp/fluent";
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

    case "scale": {
      assertNonZero(edit.factor);
      const chain = fft.forward(input).scale(edit.factor);
      return chain.inverseInto(output).real;
    }

    case "multiply_by_i": {
      const chain = fft.forward(input).mulScalar(0, 1);
      return chain.inverseInto(output).real;
    }

    case "conjugate": {
      const chain = fft.forward(input).conj();
      return chain.inverseInto(output).real;
    }

    case "negate": {
      assertNonZero(-1);
      const chain = fft.forward(input).scale(-1);
      return chain.inverseInto(output).real;
    }

    case "scale_and_conjugate": {
      assertNonZero(edit.factor);
      const chain = fft.forward(input).scale(edit.factor).conj();
      return chain.inverseInto(output).real;
    }
  }
};
