import * as Effect from "effect/Effect";
import * as Schema from "effect/Schema";
// @effect-diagnostics-next-line nodeBuiltinImport:off
import { readFileSync } from "node:fs";
// @effect-diagnostics-next-line nodeBuiltinImport:off
import { dirname, resolve } from "node:path";

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
  dirname(new URL(import.meta.url).pathname),
  "fixtures",
  "pragma-dsp.v0.1.json"
);

const loadFixturesProgram = Effect.sync(() => {
    const content = readFileSync(fixturesPath, "utf8");
    return Schema.decodeUnknownSync(Schema.UnknownFromJsonString)(
      content
    ) as Fixtures;
  });

export const loadFixtures = (): Fixtures => Effect.runSync(loadFixturesProgram);

export const getCasesByN = (fixtures: Fixtures, n: number): FixtureCase[] =>
  fixtures.fftCases.filter((c) => c.n === n);

export const getCaseByName = (
  fixtures: Fixtures,
  name: string
): FixtureCase => {
  const found = fixtures.fftCases.find((c) => c.name === name);
  if (found === undefined) {
    throw new Error(`Missing fixture case: ${name}`);
  }
  return found;
};
