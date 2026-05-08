import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import { FFT, createWindow, type WindowType } from "../xform/fourier.ts";

export interface FourierService {
  fft: (size: number) => FFT;
  window: (type: WindowType, size: number) => Float64Array;
}

export class Fourier extends Context.Service<
  Fourier,
  FourierService
>()("pragma-dsp/effect/Fourier") {}

export const FourierLive = Layer.effect(
  Fourier,
  Effect.sync(() => {
    const fftCache = new Map<number, FFT>();
    const windowCache = new Map<string, Float64Array>();

    return {
      fft: (size: number) => {
        const cached = fftCache.get(size);
        if (cached !== undefined) return cached;
        const created = new FFT(size);
        fftCache.set(size, created);
        return created;
      },
      window: (type: WindowType, size: number) => {
        const key = `${type}:${size}`;
        const cached = windowCache.get(key);
        if (cached !== undefined) return cached;
        const created = createWindow(type, size);
        windowCache.set(key, created);
        return created;
      }
    } satisfies FourierService;
  })
);
