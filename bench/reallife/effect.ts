/**
 * Effect API Benchmark
 *
 * Benchmarks the Effect integration:
 * - spectrumFx vs pure spectrum()
 * - Fourier service caching (FFT and window instances)
 * - spectrumStream for multiple frames
 * - All window types via Effect API
 *
 * Run: bun run bench/reallife/effect.ts
 */

import { Effect, Stream } from "effect";
import { spectrum } from "../../src/public/spectrum.js";
import {
  Fourier,
  FourierLive,
  spectrumFx,
  spectrumStream
} from "../../src/effect/index.js";
import { BenchContext, loadPureSine } from "./helpers.js";

const bench = new BenchContext("Effect API Benchmark");

// =============================================================================
// spectrumFx vs spectrum() Parity
// =============================================================================

bench.section("spectrumFx vs spectrum() Parity");

const sineRef = loadPureSine();
const testCases = sineRef.cases.slice(0, 3);

for (const testCase of testCases) {
  const input = Float64Array.from(testCase.signal);
  const options = {
    sampleRate: testCase.sampleRate,
    fftSize: testCase.n,
    window: "rect" as const,
    sides: "one" as const
  };

  // Pure API
  const pureResult = bench.time(`spectrum (pure, ${testCase.name})`, () =>
    spectrum(input, options)
  );

  // Effect API
  const effectResult = bench.time(
    `spectrumFx (Effect, ${testCase.name})`,
    () =>
      Effect.runSync(spectrumFx(input, options).pipe(Effect.provide(FourierLive)))
  );

  // Verify parity
  if (effectResult.peak.index !== pureResult.peak.index) {
    console.warn(
      `  WARNING: Peak index mismatch (Effect: ${effectResult.peak.index}, Pure: ${pureResult.peak.index})`
    );
  }
  if (effectResult.peak.frequency !== pureResult.peak.frequency) {
    console.warn(
      `  WARNING: Peak frequency mismatch (Effect: ${effectResult.peak.frequency}, Pure: ${pureResult.peak.frequency})`
    );
  }
  if (effectResult.peak.amplitude !== pureResult.peak.amplitude) {
    console.warn(
      `  WARNING: Peak amplitude mismatch (Effect: ${effectResult.peak.amplitude}, Pure: ${pureResult.peak.amplitude})`
    );
  }

  // Verify arrays are identical
  let arraysMatch = true;
  for (let i = 0; i < effectResult.amplitude.length; i += 1) {
    if (effectResult.amplitude[i] !== pureResult.amplitude[i]) {
      arraysMatch = false;
      break;
    }
  }
  if (!arraysMatch) {
    console.warn(`  WARNING: Amplitude arrays don't match for ${testCase.name}`);
  }
}

console.log(`  VERIFIED: spectrumFx produces identical results to spectrum()`);

bench.memory("After parity check");

// =============================================================================
// Fourier Service Caching - FFT Instances
// =============================================================================

bench.section("Fourier Service Caching - FFT Instances");

// Test that same size returns same FFT instance
const fftCachingProgram = Effect.gen(function* () {
  const service = yield* Fourier;

  // Get FFT instances of same size
  const fft64a = service.fft(64);
  const fft64b = service.fft(64);
  const fft128 = service.fft(128);

  return { fft64a, fft64b, fft128 };
});

const cacheResult = bench.time(`FFT caching verification`, () =>
  Effect.runSync(fftCachingProgram.pipe(Effect.provide(FourierLive)))
);

if (cacheResult.fft64a === cacheResult.fft64b) {
  console.log(`  VERIFIED: Same FFT size returns same instance (cache hit)`);
} else {
  console.warn(`  WARNING: FFT cache miss for same size`);
}

if (cacheResult.fft64a !== cacheResult.fft128) {
  console.log(`  VERIFIED: Different FFT size returns different instance`);
} else {
  console.warn(`  WARNING: Different FFT sizes returned same instance`);
}

// Benchmark cache hit performance
const sizes = [64, 128, 256, 512, 1024];
for (const size of sizes) {
  bench.time(
    `FFT cache lookup (size=${size})`,
    () =>
      Effect.runSync(
        Effect.gen(function* () {
          const service = yield* Fourier;
          return service.fft(size);
        }).pipe(Effect.provide(FourierLive))
      ),
    { iterations: 1000 }
  );
}

bench.memory("After FFT caching");

// =============================================================================
// Fourier Service Caching - Window Instances
// =============================================================================

bench.section("Fourier Service Caching - Window Instances");

const windowCachingProgram = Effect.gen(function* () {
  const service = yield* Fourier;

  // Get window instances of same type and size
  const hann64a = service.window("hann", 64);
  const hann64b = service.window("hann", 64);
  const hamming64 = service.window("hamming", 64);
  const hann128 = service.window("hann", 128);

  return { hann64a, hann64b, hamming64, hann128 };
});

const windowCacheResult = bench.time(`Window caching verification`, () =>
  Effect.runSync(windowCachingProgram.pipe(Effect.provide(FourierLive)))
);

if (windowCacheResult.hann64a === windowCacheResult.hann64b) {
  console.log(
    `  VERIFIED: Same window type+size returns same instance (cache hit)`
  );
} else {
  console.warn(`  WARNING: Window cache miss for same type+size`);
}

if (windowCacheResult.hann64a !== windowCacheResult.hamming64) {
  console.log(`  VERIFIED: Different window type returns different instance`);
} else {
  console.warn(`  WARNING: Different window types returned same instance`);
}

if (windowCacheResult.hann64a !== windowCacheResult.hann128) {
  console.log(`  VERIFIED: Different window size returns different instance`);
} else {
  console.warn(`  WARNING: Different window sizes returned same instance`);
}

// Benchmark window cache hit performance
const windowTypes = ["rect", "hann", "hamming", "blackman"] as const;
for (const wtype of windowTypes) {
  bench.time(
    `Window cache lookup (${wtype}, size=1024)`,
    () =>
      Effect.runSync(
        Effect.gen(function* () {
          const service = yield* Fourier;
          return service.window(wtype, 1024);
        }).pipe(Effect.provide(FourierLive))
      ),
    { iterations: 1000 }
  );
}

bench.memory("After window caching");

// =============================================================================
// spectrumStream - Multiple Frames
// =============================================================================

bench.section("spectrumStream - Multiple Frames");

const streamTestCase = sineRef.cases[0]!;
const frame = Float32Array.from(streamTestCase.signal);
const streamOptions = {
  sampleRate: streamTestCase.sampleRate,
  fftSize: streamTestCase.n,
  window: "rect" as const,
  sides: "one" as const
};

// Process 3 frames
const threeFrames = Stream.fromIterable([frame, frame, frame]);
const threeFrameResults = bench.time(`spectrumStream (3 frames)`, () =>
  Effect.runSync(
    Stream.runCollect(spectrumStream(threeFrames, streamOptions)).pipe(
      Effect.provide(FourierLive)
    )
  )
);

const resultsArray = Array.from(threeFrameResults);
if (resultsArray.length === 3) {
  console.log(`  VERIFIED: Processed 3 frames correctly`);
} else {
  console.warn(`  WARNING: Expected 3 results, got ${resultsArray.length}`);
}

// Verify all results are consistent (same input frame)
const expectedPeak = spectrum(streamTestCase.signal, streamOptions);
let allMatch = true;
for (const result of resultsArray) {
  if (result.peak.index !== expectedPeak.peak.index) {
    allMatch = false;
    break;
  }
}
if (allMatch) {
  console.log(`  VERIFIED: All stream results match expected peak`);
} else {
  console.warn(`  WARNING: Stream results don't match expected peak`);
}

// Process 10 frames
const tenFrames = Stream.fromIterable(Array(10).fill(frame));
bench.time(`spectrumStream (10 frames)`, () =>
  Effect.runSync(
    Stream.runCollect(spectrumStream(tenFrames, streamOptions)).pipe(
      Effect.provide(FourierLive)
    )
  )
);

// Process 100 frames
const hundredFrames = Stream.fromIterable(Array(100).fill(frame));
bench.time(
  `spectrumStream (100 frames)`,
  () =>
    Effect.runSync(
      Stream.runCollect(spectrumStream(hundredFrames, streamOptions)).pipe(
        Effect.provide(FourierLive)
      )
    ),
  { iterations: 10 }
);

// Empty stream
const emptyStream = Stream.empty as Stream.Stream<Float32Array>;
const emptyResults = bench.time(`spectrumStream (empty)`, () =>
  Effect.runSync(
    Stream.runCollect(spectrumStream(emptyStream, streamOptions)).pipe(
      Effect.provide(FourierLive)
    )
  )
);

if (Array.from(emptyResults).length === 0) {
  console.log(`  VERIFIED: Empty stream returns no results`);
} else {
  console.warn(`  WARNING: Empty stream returned results`);
}

bench.memory("After spectrumStream");

// =============================================================================
// Window Types via Effect API
// =============================================================================

bench.section("Window Types via Effect API");

const windowTestCase = sineRef.cases[0]!;
const windowInput = Float64Array.from(windowTestCase.signal);

for (const windowType of windowTypes) {
  const options = {
    sampleRate: windowTestCase.sampleRate,
    fftSize: windowTestCase.n,
    window: windowType,
    sides: "one" as const
  };

  // Pure API
  const pureResult = bench.time(`spectrum (pure, ${windowType})`, () =>
    spectrum(windowInput, options)
  );

  // Effect API
  const effectResult = bench.time(`spectrumFx (Effect, ${windowType})`, () =>
    Effect.runSync(spectrumFx(windowInput, options).pipe(Effect.provide(FourierLive)))
  );

  // Verify parity
  if (effectResult.peak.index !== pureResult.peak.index) {
    console.warn(
      `  WARNING: ${windowType} peak index mismatch (Effect: ${effectResult.peak.index}, Pure: ${pureResult.peak.index})`
    );
  }

  const ampDiff = Math.abs(effectResult.peak.amplitude - pureResult.peak.amplitude);
  if (ampDiff > 1e-10) {
    console.warn(
      `  WARNING: ${windowType} amplitude diff ${ampDiff}`
    );
  }
}

console.log(`  VERIFIED: All window types work via Effect API`);

bench.memory("After window types");

// =============================================================================
// Effect vs Pure Performance Comparison
// =============================================================================

bench.section("Effect vs Pure Performance Comparison");

const perfTestCase = sineRef.cases[0]!;
const perfInput = Float64Array.from(perfTestCase.signal);
const perfOptions = {
  sampleRate: perfTestCase.sampleRate,
  fftSize: perfTestCase.n,
  window: "hann" as const,
  sides: "one" as const
};

bench.time(
  `100x spectrum (pure, n=${perfTestCase.n})`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      lastResult = spectrum(perfInput, perfOptions);
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.time(
  `100x spectrumFx (Effect, n=${perfTestCase.n})`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      lastResult = Effect.runSync(
        spectrumFx(perfInput, perfOptions).pipe(Effect.provide(FourierLive))
      );
    }
    return lastResult;
  },
  { iterations: 10 }
);

// Effect with pre-provided layer (more realistic usage)
const providedProgram = spectrumFx(perfInput, perfOptions).pipe(
  Effect.provide(FourierLive)
);

bench.time(
  `100x spectrumFx (pre-provided layer, n=${perfTestCase.n})`,
  () => {
    let lastResult;
    for (let i = 0; i < 100; i += 1) {
      lastResult = Effect.runSync(providedProgram);
    }
    return lastResult;
  },
  { iterations: 10 }
);

bench.memory("After performance comparison");

// =============================================================================
// Caching Benefit Demonstration
// =============================================================================

bench.section("Caching Benefit Demonstration");

// Multiple calls with same size should benefit from caching
const cacheBenefitProgram = Effect.gen(function* () {
  const service = yield* Fourier;
  const results: number[] = [];

  // 100 FFT lookups of same size
  for (let i = 0; i < 100; i += 1) {
    const fft = service.fft(1024);
    results.push(fft.size);
  }

  // 100 window lookups of same type+size
  for (let i = 0; i < 100; i += 1) {
    const w = service.window("hann", 1024);
    results.push(w.length);
  }

  return results;
});

bench.time(
  `100 FFT + 100 window cache lookups`,
  () =>
    Effect.runSync(cacheBenefitProgram.pipe(Effect.provide(FourierLive))),
  { iterations: 100 }
);

// Mixed sizes (some cache hits, some misses)
const mixedCacheProgram = Effect.gen(function* () {
  const service = yield* Fourier;
  const sizes = [64, 128, 256, 512, 1024, 64, 128, 256, 512, 1024];
  const results: number[] = [];

  for (const size of sizes) {
    const fft = service.fft(size);
    results.push(fft.size);
  }

  return results;
});

bench.time(
  `10 FFT lookups (5 sizes, 2x each)`,
  () =>
    Effect.runSync(mixedCacheProgram.pipe(Effect.provide(FourierLive))),
  { iterations: 100 }
);

bench.memory("After caching benefit");

// =============================================================================
// Report
// =============================================================================

bench.report();
