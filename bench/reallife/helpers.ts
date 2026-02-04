/**
 * Shared utilities for real-life benchmarks.
 * Provides reference loaders and BenchContext for measuring performance.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const refsDir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../test/reallife/references"
);

// =============================================================================
// Reference types (same as test/reallife/helpers.ts)
// =============================================================================

export type SignalCase = {
  name: string;
  kind: string;
  n: number;
  sampleRate: number;
  signal: number[];
  fftRe: number[];
  fftIm: number[];
  magnitude: number[];
  phase: number[];
  peakBin: number;
  peakMagnitude: number;
  peakPhase: number;
  params: Record<string, unknown>;
};

export type SignalReference = {
  generatedAt: string;
  generator: string;
  python: string;
  numpy: string;
  scipy: string;
  platform: string;
  description: string;
  n: number;
  sampleRate: number;
  cases: SignalCase[];
};

export type WindowDspCase = {
  type: "rect" | "hann" | "hamming" | "blackman";
  n: number;
  values: number[];
  coherentGain: number;
  enbw: number;
};

export type WindowDspReference = {
  generatedAt: string;
  generator: string;
  python: string;
  numpy: string;
  scipy: string;
  platform: string;
  description: string;
  cases: WindowDspCase[];
};

// =============================================================================
// Reference loaders
// =============================================================================

export const loadReference = <T>(name: string): T => {
  const path = resolve(refsDir, `${name}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as T;
};

export const loadPureSine = (): SignalReference =>
  loadReference<SignalReference>("pure_sine");

export const loadCosine = (): SignalReference =>
  loadReference<SignalReference>("cosine");

export const loadMultiTone = (): SignalReference =>
  loadReference<SignalReference>("multi_tone");

export const loadChirp = (): SignalReference =>
  loadReference<SignalReference>("chirp");

export const loadSpecial = (): SignalReference =>
  loadReference<SignalReference>("special");

export const loadWindowsDsp = (): WindowDspReference =>
  loadReference<WindowDspReference>("windows_dsp");

// =============================================================================
// Utility functions
// =============================================================================

/**
 * Compute the maximum absolute error between two arrays.
 */
export const maxAbsError = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>
): number => {
  let max = 0;
  const len = Math.min(actual.length, expected.length);
  for (let i = 0; i < len; i += 1) {
    const diff = Math.abs((actual[i] ?? 0) - (expected[i] ?? 0));
    if (diff > max) max = diff;
  }
  return max;
};

/**
 * Compute a simple checksum to prevent dead-code elimination.
 */
export const checksum = (arr: ArrayLike<number>): number => {
  let sum = 0;
  for (let i = 0; i < arr.length; i += 1) {
    sum += (arr[i] ?? 0) * (i + 1) * 0.001;
  }
  return sum;
};

// =============================================================================
// BenchContext - Benchmark utility class
// =============================================================================

type TimingResult = {
  label: string;
  iterations: number;
  totalMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  checksum: number;
};

type MemorySnapshot = {
  label: string;
  heapUsed: number;
  heapTotal: number;
  rss: number;
};

type SectionResult = {
  name: string;
  timings: TimingResult[];
  memory: MemorySnapshot[];
};

export class BenchContext {
  private sections: SectionResult[] = [];
  private currentSection: SectionResult | null = null;
  private startTime: number;
  private readonly title: string;

  constructor(title: string) {
    this.title = title;
    this.startTime = performance.now();
  }

  /**
   * Start a new section of benchmarks.
   */
  section(name: string): void {
    this.currentSection = { name, timings: [], memory: [] };
    this.sections.push(this.currentSection);
  }

  /**
   * Force garbage collection if available (Bun.gc).
   */
  gc(): void {
    if (typeof Bun !== "undefined" && typeof Bun.gc === "function") {
      Bun.gc(true);
    } else if (typeof globalThis.gc === "function") {
      globalThis.gc();
    }
  }

  /**
   * Measure execution time for a function.
   */
  time<T>(
    label: string,
    fn: () => T,
    options: { iterations?: number; warmup?: number } = {}
  ): T {
    const iterations = options.iterations ?? 100;
    const warmup = options.warmup ?? 10;

    // Warmup runs
    let result: T;
    for (let i = 0; i < warmup; i += 1) {
      result = fn();
    }

    // Force GC before measurement
    this.gc();

    // Timed runs
    const times: number[] = [];
    let checksumValue = 0;

    for (let i = 0; i < iterations; i += 1) {
      const start = performance.now();
      result = fn();
      const end = performance.now();
      times.push(end - start);

      // Compute checksum to prevent dead-code elimination
      if (result && typeof result === "object") {
        if ("real" in result && ArrayBuffer.isView((result as any).real)) {
          checksumValue += checksum((result as any).real);
        } else if (ArrayBuffer.isView(result)) {
          checksumValue += checksum(result as any);
        } else if ("amplitude" in result && ArrayBuffer.isView((result as any).amplitude)) {
          checksumValue += checksum((result as any).amplitude);
        }
      }
    }

    const totalMs = times.reduce((a, b) => a + b, 0);
    const avgMs = totalMs / iterations;
    const minMs = Math.min(...times);
    const maxMs = Math.max(...times);

    const timing: TimingResult = {
      label,
      iterations,
      totalMs,
      avgMs,
      minMs,
      maxMs,
      checksum: checksumValue
    };

    if (this.currentSection) {
      this.currentSection.timings.push(timing);
    }

    return result!;
  }

  /**
   * Capture a memory snapshot.
   */
  memory(label: string): MemorySnapshot {
    this.gc();

    const mem = process.memoryUsage();
    const snapshot: MemorySnapshot = {
      label,
      heapUsed: mem.heapUsed,
      heapTotal: mem.heapTotal,
      rss: mem.rss
    };

    if (this.currentSection) {
      this.currentSection.memory.push(snapshot);
    }

    return snapshot;
  }

  /**
   * Format bytes to human-readable string.
   */
  private formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
  }

  /**
   * Format milliseconds to human-readable string.
   */
  private formatMs(ms: number): string {
    if (ms < 0.001) return `${(ms * 1000000).toFixed(0)}ns`;
    if (ms < 1) return `${(ms * 1000).toFixed(1)}us`;
    if (ms < 1000) return `${ms.toFixed(3)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  /**
   * Output the benchmark report to stdout.
   */
  report(): void {
    const totalTime = performance.now() - this.startTime;
    const divider = "=".repeat(70);
    const thinDivider = "-".repeat(70);

    console.log();
    console.log(divider);
    console.log(`  ${this.title.toUpperCase()}`);
    console.log(`  ${new Date().toISOString()}`);
    console.log(divider);
    console.log();

    let totalChecksum = 0;

    for (const section of this.sections) {
      console.log(`[${section.name}]`);
      console.log();

      // Print timings
      for (const t of section.timings) {
        const label = t.label.padEnd(45);
        const iters = `x${t.iterations}`.padStart(6);
        const avg = `avg: ${this.formatMs(t.avgMs)}`.padStart(14);
        const min = `min: ${this.formatMs(t.minMs)}`.padStart(14);
        const max = `max: ${this.formatMs(t.maxMs)}`.padStart(14);
        console.log(`  ${label} ${iters} ${avg} ${min} ${max}`);
        totalChecksum += t.checksum;
      }

      // Print memory snapshots
      if (section.memory.length > 0) {
        console.log();
        for (const m of section.memory) {
          const label = m.label.padEnd(45);
          const heap = `heap: ${this.formatBytes(m.heapUsed)}`.padStart(16);
          const rss = `rss: ${this.formatBytes(m.rss)}`.padStart(16);
          console.log(`  ${label} ${heap} ${rss}`);
        }
      }

      console.log();
    }

    console.log(thinDivider);
    console.log(`  Total benchmark time: ${this.formatMs(totalTime)}`);
    console.log(`  Checksum (for validation): ${totalChecksum.toFixed(6)}`);
    console.log(thinDivider);
    console.log();
  }
}

// =============================================================================
// Declare Bun global for TypeScript
// =============================================================================

declare global {
  var Bun: { gc: (full?: boolean) => void } | undefined;
  var gc: (() => void) | undefined;
}
