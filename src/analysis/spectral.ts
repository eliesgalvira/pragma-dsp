export type MovingAverageOptions = {
  /** Width of the smoothing window (samples). */
  windowSize?: number;
};

export const movingAverage = (
  input: ArrayLike<number>,
  windowSize = 11,
  out?: Float64Array
): Float64Array => {
  if (windowSize <= 0) {
    throw new Error(`Window size must be positive, got ${windowSize}`);
  }
  const size = input.length;
  const result = out ?? new Float64Array(size);
  const half = Math.floor(windowSize / 2);

  for (let i = 0; i < size; i += 1) {
    let sum = 0;
    let count = 0;
    const start = Math.max(0, i - half);
    const end = Math.min(size - 1, i + half);
    for (let j = start; j <= end; j += 1) {
      sum += input[j] ?? 0;
      count += 1;
    }
    result[i] = count > 0 ? sum / count : 0;
  }

  return result;
};

export type SpectralEnvelopeOptions = {
  /** Smoothing window width (samples). */
  smoothingWidth?: number;
};

/**
 * Smooth a magnitude spectrum into a simple spectral envelope.
 */
export const spectralEnvelope = (
  magnitudes: ArrayLike<number>,
  options: SpectralEnvelopeOptions = {},
  out?: Float64Array
): Float64Array => {
  const width = options.smoothingWidth ?? 11;
  return movingAverage(magnitudes, width, out);
};

export type MagnitudeToDbOptions = {
  /** Minimum dB floor (clamp). */
  floorDb?: number;
};

/**
 * Convert linear magnitudes to dB (20 * log10), clamped to a floor.
 */
export const magnitudeToDb = (
  magnitudes: ArrayLike<number>,
  options: MagnitudeToDbOptions = {},
  out?: Float64Array
): Float64Array => {
  const floorDb = options.floorDb ?? -80;
  const result = out ?? new Float64Array(magnitudes.length);
  for (let i = 0; i < magnitudes.length; i += 1) {
    const val = magnitudes[i] ?? 0;
    const db = val > 0 ? 20 * Math.log10(val) : floorDb;
    result[i] = db < floorDb ? floorDb : db;
  }
  return result;
};
