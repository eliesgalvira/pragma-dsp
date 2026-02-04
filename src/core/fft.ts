export type ComplexArray = {
  real: Float64Array;
  imag: Float64Array;
};

export const createComplexArray = (size: number, fill = 0): ComplexArray => {
  const real = new Float64Array(size);
  const imag = new Float64Array(size);
  if (fill !== 0) {
    real.fill(fill);
    imag.fill(fill);
  }
  return { real, imag };
};

export const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0;

export const nextPowerOfTwo = (n: number): number => {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
};

const buildBitReverse = (size: number): Uint32Array => {
  const bits = Math.round(Math.log2(size));
  const rev = new Uint32Array(size);
  for (let i = 0; i < size; i += 1) {
    let x = i;
    let y = 0;
    for (let b = 0; b < bits; b += 1) {
      y = (y << 1) | (x & 1);
      x >>= 1;
    }
    rev[i] = y;
  }
  return rev;
};

type TwiddleTable = {
  cos: Float64Array;
  sin: Float64Array;
};

const buildTwiddles = (size: number): TwiddleTable[] => {
  const stages = Math.round(Math.log2(size));
  const tables: TwiddleTable[] = [];
  for (let stage = 1; stage <= stages; stage += 1) {
    const m = 1 << stage;
    const half = m >> 1;
    const cos = new Float64Array(half);
    const sin = new Float64Array(half);
    for (let k = 0; k < half; k += 1) {
      const angle = (-2 * Math.PI * k) / m;
      cos[k] = Math.cos(angle);
      sin[k] = Math.sin(angle);
    }
    tables.push({ cos, sin });
  }
  return tables;
};

export class Radix2Fft {
  readonly size: number;
  private readonly bitReverse: Uint32Array;
  private readonly twiddles: TwiddleTable[];

  constructor(size: number) {
    if (!isPowerOfTwo(size)) {
      throw new Error(`FFT size must be power of two, got ${size}`);
    }
    this.size = size;
    this.bitReverse = buildBitReverse(size);
    this.twiddles = buildTwiddles(size);
  }

  forward(input: ArrayLike<number>, out?: ComplexArray): ComplexArray {
    return this.transform(input, null, out, false);
  }

  forwardComplex(input: ComplexArray, out?: ComplexArray): ComplexArray {
    return this.transform(input.real, input.imag, out, false);
  }

  inverse(input: ComplexArray, out?: ComplexArray): ComplexArray {
    return this.transform(input.real, input.imag, out, true);
  }

  private transform(
    inputReal: ArrayLike<number>,
    inputImag: ArrayLike<number> | null,
    out: ComplexArray | undefined,
    inverse: boolean
  ): ComplexArray {
    if (inputReal.length !== this.size) {
      throw new Error(
        `FFT input length ${inputReal.length} != size ${this.size}`
      );
    }
    if (inputImag && inputImag.length !== this.size) {
      throw new Error(
        `FFT input length ${inputImag.length} != size ${this.size}`
      );
    }

    const result = out ?? createComplexArray(this.size);
    const outReal = result.real;
    const outImag = result.imag;

    for (let i = 0; i < this.size; i += 1) {
      const j = this.bitReverse[i]!;
      outReal[j] = inputReal[i] ?? 0;
      outImag[j] = inputImag ? inputImag[i] ?? 0 : 0;
    }

    for (let stage = 0; stage < this.twiddles.length; stage += 1) {
      const table = this.twiddles[stage]!;
      const m = 1 << (stage + 1);
      const half = m >> 1;
      const cos = table.cos;
      const sin = table.sin;
      const sinSign = inverse ? -1 : 1;

      for (let k = 0; k < this.size; k += m) {
        for (let j = 0; j < half; j += 1) {
          const tReal =
            cos[j]! * outReal[k + j + half]! -
            sinSign * sin[j]! * outImag[k + j + half]!;
          const tImag =
            sinSign * sin[j]! * outReal[k + j + half]! +
            cos[j]! * outImag[k + j + half]!;
          const uReal = outReal[k + j]!;
          const uImag = outImag[k + j]!;
          outReal[k + j] = uReal + tReal;
          outImag[k + j] = uImag + tImag;
          outReal[k + j + half] = uReal - tReal;
          outImag[k + j + half] = uImag - tImag;
        }
      }
    }

    if (inverse) {
      const scale = 1 / this.size;
      for (let i = 0; i < this.size; i += 1) {
        outReal[i] = outReal[i]! * scale;
        outImag[i] = outImag[i]! * scale;
      }
    }

    return result;
  }
}
