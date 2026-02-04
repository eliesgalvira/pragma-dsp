import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type FixtureWindow = {
  type: "rect" | "hann" | "hamming" | "blackman";
  n: number;
  sym: boolean;
  values: number[];
};

export type FixtureCase = {
  name: string;
  kind: string;
  n: number;
  sampleRate: number;
  input: number[];
  fftRe: number[];
  fftIm: number[];
  meta: Record<string, number>;
};

export type Fixtures = {
  schemaVersion: "0.1";
  generatedAt: string;
  generator: {
    tool: string;
    seed: number;
    python: string;
    numpy: string;
    scipy: string | null;
    platform: string;
  };
  convention: {
    forward: string;
    inverse: string;
    normalization: string;
    note?: string;
  };
  windows: FixtureWindow[];
  fftCases: FixtureCase[];
};

const fixturesPath = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "pragma-dsp.v0.1.json"
);

export const loadFixtures = (): Fixtures =>
  JSON.parse(readFileSync(fixturesPath, "utf8")) as Fixtures;

export const getCasesByN = (fixtures: Fixtures, n: number): FixtureCase[] =>
  fixtures.fftCases.filter((c) => c.n === n);

export const getCaseByName = (
  fixtures: Fixtures,
  name: string
): FixtureCase => {
  const found = fixtures.fftCases.find((c) => c.name === name);
  if (!found) {
    throw new Error(`Missing fixture case: ${name}`);
  }
  return found;
};
