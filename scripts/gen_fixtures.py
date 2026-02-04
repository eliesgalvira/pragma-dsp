#!/usr/bin/env python3
"""
Generate pragma-dsp v0.1 fixtures.

Goals:
- Provide trusted reference FFT outputs from NumPy for correctness tests.
- Provide stable, deterministic inputs for benchmarks (JSON-driven).
- Optionally generate window arrays; uses SciPy if available, otherwise uses
  explicit formulae matching typical DSP definitions (sym=True).

DFT convention (matches numpy.fft.fft):
- Forward:  X[k] = sum_{n=0..N-1} x[n] * exp(-j * 2*pi*k*n/N)
- Inverse:  x[n] = (1/N) * sum_{k=0..N-1} X[k] * exp(+j * 2*pi*k*n/N)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def try_import_scipy_windows():
  try:
    from scipy.signal import windows as sp_windows  # type: ignore
    return sp_windows
  except Exception:
    return None


def window_rect(n: int) -> np.ndarray:
  return np.ones(n, dtype=np.float64)


def window_hann_sym(n: int) -> np.ndarray:
  # Hann (a.k.a. Hanning) symmetric:
  # w[i] = 0.5 * (1 - cos(2*pi*i/(N-1)))
  if n <= 1:
    return np.ones(n, dtype=np.float64)
  i = np.arange(n, dtype=np.float64)
  return 0.5 * (1.0 - np.cos((2.0 * np.pi * i) / (n - 1)))


def window_hamming_sym(n: int) -> np.ndarray:
  # Hamming symmetric:
  # w[i] = 0.54 - 0.46*cos(2*pi*i/(N-1))
  if n <= 1:
    return np.ones(n, dtype=np.float64)
  i = np.arange(n, dtype=np.float64)
  return 0.54 - 0.46 * np.cos((2.0 * np.pi * i) / (n - 1))


def window_blackman_sym(n: int) -> np.ndarray:
  # Blackman symmetric:
  # w[i] = a0 - a1*cos(2*pi*i/(N-1)) + a2*cos(4*pi*i/(N-1))
  if n <= 1:
    return np.ones(n, dtype=np.float64)
  a0, a1, a2 = 0.42, 0.5, 0.08
  i = np.arange(n, dtype=np.float64)
  f = (2.0 * np.pi * i) / (n - 1)
  return a0 - a1 * np.cos(f) + a2 * np.cos(2.0 * f)


WINDOW_FORMULAS = {
  "rect": window_rect,
  "hann": window_hann_sym,
  "hamming": window_hamming_sym,
  "blackman": window_blackman_sym,
}


def generate_bin_centered_sine(
  n: int, k: int, amplitude: float, phase: float = 0.0
) -> np.ndarray:
  # x[n] = A * sin(2*pi*k*n/N - phase)
  idx = np.arange(n, dtype=np.float64)
  return amplitude * np.sin((2.0 * np.pi * k * idx) / n - phase)


def fft_reference(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  X = np.fft.fft(x.astype(np.float64))
  return X.real.astype(np.float64), X.imag.astype(np.float64)


def to_float_list(arr: np.ndarray) -> List[float]:
  return [float(x) for x in arr.tolist()]


def utc_now_iso() -> str:
  return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclasses.dataclass(frozen=True)
class FFTCaseSpec:
  name: str
  n: int
  sample_rate: float
  input: np.ndarray
  kind: str


def build_cases(
  rng: np.random.Generator,
  *,
  small_sizes: List[int],
  small_cases_per_size: int,
  bench_sizes: List[int],
  sample_rate: float,
  sine_case: Tuple[int, int, float],
) -> List[FFTCaseSpec]:
  cases: List[FFTCaseSpec] = []

  # Random small cases for correctness against DFT/NumPy FFT.
  for n in small_sizes:
    for i in range(small_cases_per_size):
      x = rng.standard_normal(n).astype(np.float64)
      cases.append(
        FFTCaseSpec(
          name=f"rand_n{n}_{i}",
          n=n,
          sample_rate=sample_rate,
          input=x,
          kind="random_normal",
        )
      )

  # Deterministic sine case for amplitude scaling and peak detection tests.
  # sine_case = (N, k, amplitude)
  sine_n, sine_k, sine_amp = sine_case
  x_sine = generate_bin_centered_sine(sine_n, sine_k, sine_amp, phase=0.0)
  cases.append(
    FFTCaseSpec(
      name=f"sine_bincentered_n{sine_n}_k{sine_k}_a{sine_amp}",
      n=sine_n,
      sample_rate=sample_rate,
      input=x_sine,
      kind="sine_bin_centered",
    )
  )

  # Benchmark cases: 1 random per bench size (stable inputs).
  for n in bench_sizes:
    x = rng.standard_normal(n).astype(np.float64)
    cases.append(
      FFTCaseSpec(
        name=f"bench_rand_n{n}",
        n=n,
        sample_rate=sample_rate,
        input=x,
        kind="benchmark_random_normal",
      )
    )

  return cases


def build_windows(
  *,
  window_types: List[str],
  sizes: List[int],
) -> List[Dict[str, Any]]:
  sp_windows = try_import_scipy_windows()
  out: List[Dict[str, Any]] = []

  for wtype in window_types:
    if wtype not in WINDOW_FORMULAS:
      raise ValueError(f"Unknown window type: {wtype}")

    for n in sizes:
      # Prefer SciPy if present so we align with a canonical implementation,
      # but fall back to our formula (sym=True equivalent) if SciPy is absent.
      if sp_windows is not None:
        if wtype == "rect":
          w = np.ones(n, dtype=np.float64)
        elif wtype == "hann":
          w = sp_windows.hann(n, sym=True).astype(np.float64)
        elif wtype == "hamming":
          w = sp_windows.hamming(n, sym=True).astype(np.float64)
        elif wtype == "blackman":
          w = sp_windows.blackman(n, sym=True).astype(np.float64)
        else:
          w = WINDOW_FORMULAS[wtype](n)
      else:
        w = WINDOW_FORMULAS[wtype](n)

      out.append(
        {
          "type": wtype,
          "n": int(n),
          "sym": True,
          "values": to_float_list(w),
        }
      )

  return out


def ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument(
    "--out",
    default="test/fixtures/pragma-dsp.v0.1.json",
    help="Output JSON fixture path",
  )
  p.add_argument(
    "--seed",
    type=int,
    default=1337,
    help="Deterministic RNG seed",
  )
  p.add_argument(
    "--sample-rate",
    type=float,
    default=48000.0,
    help="Sample rate metadata for cases (Hz)",
  )
  p.add_argument(
    "--small-sizes",
    default="8,16,32",
    help="Comma-separated FFT sizes for correctness fixtures",
  )
  p.add_argument(
    "--small-cases-per-size",
    type=int,
    default=5,
    help="How many random cases per small size",
  )
  p.add_argument(
    "--bench-sizes",
    default="2048,4096",
    help="Comma-separated FFT sizes for benchmark fixtures",
  )
  p.add_argument(
    "--window-sizes",
    default="8,16,32,64,1024,2048,4096",
    help="Comma-separated sizes for window fixtures",
  )
  p.add_argument(
    "--windows",
    default="rect,hann,hamming,blackman",
    help="Comma-separated window types to generate",
  )
  p.add_argument(
    "--sine-n",
    type=int,
    default=1024,
    help="N for the bin-centered sine fixture",
  )
  p.add_argument(
    "--sine-k",
    type=int,
    default=32,
    help="Bin index k for the bin-centered sine fixture",
  )
  p.add_argument(
    "--sine-amp",
    type=float,
    default=0.8,
    help="Amplitude for the bin-centered sine fixture",
  )
  p.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite output file if it exists",
  )
  args = p.parse_args()

  out_path = args.out
  if os.path.exists(out_path) and not args.overwrite:
    print(f"Refusing to overwrite existing file: {out_path}", file=sys.stderr)
    print("Pass --overwrite to regenerate fixtures.", file=sys.stderr)
    return 2

  small_sizes = [int(x) for x in args.small_sizes.split(",") if x.strip()]
  bench_sizes = [int(x) for x in args.bench_sizes.split(",") if x.strip()]
  window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]
  window_types = [x.strip() for x in args.windows.split(",") if x.strip()]

  rng = np.random.default_rng(args.seed)

  cases = build_cases(
    rng,
    small_sizes=small_sizes,
    small_cases_per_size=args.small_cases_per_size,
    bench_sizes=bench_sizes,
    sample_rate=args.sample_rate,
    sine_case=(args.sine_n, args.sine_k, args.sine_amp),
  )

  fft_cases: List[Dict[str, Any]] = []
  for c in cases:
    re, im = fft_reference(c.input)
    fft_cases.append(
      {
        "name": c.name,
        "kind": c.kind,
        "n": int(c.n),
        "sampleRate": float(c.sample_rate),
        "input": to_float_list(c.input),
        "fftRe": to_float_list(re),
        "fftIm": to_float_list(im),
        "meta": (
          {
            "binCenteredK": int(args.sine_k),
            "expectedPeakHz": float(args.sine_k * args.sample_rate / args.sine_n),
            "amplitude": float(args.sine_amp),
          }
          if c.kind == "sine_bin_centered"
          else {}
        ),
      }
    )

  windows = build_windows(window_types=window_types, sizes=window_sizes)

  sp_windows = try_import_scipy_windows()
  scipy_version: Optional[str] = None
  if sp_windows is not None:
    try:
      import scipy  # type: ignore

      scipy_version = getattr(scipy, "__version__", None)
    except Exception:
      scipy_version = "unknown"

  payload: Dict[str, Any] = {
    "schemaVersion": "0.1",
    "generatedAt": utc_now_iso(),
    "generator": {
      "tool": "scripts/gen_fixtures.py",
      "seed": int(args.seed),
      "python": platform.python_version(),
      "numpy": getattr(np, "__version__", "unknown"),
      "scipy": scipy_version,
      "platform": platform.platform(),
    },
    "convention": {
      "forward": "X[k] = sum_{n=0..N-1} x[n] * exp(-j*2*pi*k*n/N)",
      "inverse": "x[n] = (1/N) * sum_{k=0..N-1} X[k] * exp(+j*2*pi*k*n/N)",
      "normalization": "none",
      "note": "Matches numpy.fft.fft and numpy.fft.ifft conventions.",
    },
    "windows": windows,
    "fftCases": fft_cases,
  }

  ensure_dir(os.path.dirname(out_path) or ".")
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=False)
    f.write("\n")

  print(f"Wrote fixtures: {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
