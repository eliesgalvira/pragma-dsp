import { describe, expect, it } from "vitest";
import {
  magnitudeToDb,
  movingAverage,
  spectralEnvelope,
} from "../../src/analysis/spectral.js";

describe("spectral helpers", () => {
  it("movingAverage smooths with edge handling", () => {
    const input = Float64Array.from([1, 2, 3, 4, 5]);
    const out = movingAverage(input, 3);
    expect(Array.from(out)).toEqual([1.5, 2, 3, 4, 4.5]);
  });

  it("spectralEnvelope delegates to movingAverage", () => {
    const input = Float64Array.from([1, 2, 3, 4, 5]);
    const out = spectralEnvelope(input, { smoothingWidth: 3 });
    expect(Array.from(out)).toEqual([1.5, 2, 3, 4, 4.5]);
  });

  it("magnitudeToDb converts and clamps", () => {
    const input = Float64Array.from([1, 0.1, 0, 10]);
    const out = magnitudeToDb(input, { floorDb: -40 });
    expect(out[0]).toBeCloseTo(0, 8);
    expect(out[1]).toBeCloseTo(-20, 8);
    expect(out[2]).toBe(-40);
    expect(out[3]).toBeCloseTo(20, 8);
  });
});
