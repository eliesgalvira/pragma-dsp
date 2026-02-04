import { describe, expect, it } from "vitest";
import { createWindow } from "../src/xform/fourier.js";
import { loadFixtures } from "./fixtures.js";

const expectCloseArray = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tol = 1e-8
) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    const a = actual[i] ?? 0;
    const e = expected[i] ?? 0;
    expect(Math.abs(a - e)).toBeLessThanOrEqual(tol);
  }
};

describe("window fixtures", () => {
  const fixtures = loadFixtures();

  for (const window of fixtures.windows) {
    it(`matches ${window.type} window n=${window.n}`, () => {
      const created = createWindow(window.type, window.n);
      expectCloseArray(created, window.values);
    });
  }
});
