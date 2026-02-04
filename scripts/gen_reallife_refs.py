#!/usr/bin/env python3
"""
Generate real-life test references for pragma-dsp.

This script generates comprehensive test cases with NumPy/SciPy as the
reference implementation. The output JSON files are used by TypeScript
tests to validate pragma-dsp against trusted Python implementations.

Usage:
    uv run scripts/gen_reallife_refs.py --out test/reallife/references/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import windows as sp_windows


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def to_float_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in arr.tolist()]


def generator_meta() -> dict[str, Any]:
    import scipy

    return {
        "generatedAt": utc_now_iso(),
        "generator": "scripts/gen_reallife_refs.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


# =============================================================================
# Signal generators
# =============================================================================


def generate_sine(
    freq_hz: float,
    amplitude: float,
    phase_rad: float,
    sample_rate: float,
    n: int,
) -> np.ndarray:
    """Generate a sine wave: A * sin(2*pi*f*t + phase)"""
    t = np.arange(n, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase_rad)


def generate_cosine(
    freq_hz: float,
    amplitude: float,
    phase_rad: float,
    sample_rate: float,
    n: int,
) -> np.ndarray:
    """Generate a cosine wave: A * cos(2*pi*f*t + phase)"""
    t = np.arange(n, dtype=np.float64) / sample_rate
    return amplitude * np.cos(2 * np.pi * freq_hz * t + phase_rad)


def generate_multi_tone(
    freqs_hz: list[float],
    amplitudes: list[float],
    phases_rad: list[float],
    sample_rate: float,
    n: int,
) -> np.ndarray:
    """Generate sum of multiple sine waves."""
    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = np.zeros(n, dtype=np.float64)
    for freq, amp, phase in zip(freqs_hz, amplitudes, phases_rad):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal


def generate_chirp(
    f0_hz: float,
    f1_hz: float,
    sample_rate: float,
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a linear chirp from f0 to f1."""
    t = np.arange(n, dtype=np.float64) / sample_rate
    duration = n / sample_rate
    # Linear chirp: f(t) = f0 + (f1 - f0) * t / T
    # phase = 2*pi * integral of f(t) = 2*pi * (f0*t + (f1-f0)*t^2/(2*T))
    phase = 2 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * t**2 / (2 * duration))
    return amplitude * np.sin(phase)


def generate_impulse(n: int, position: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """Generate an impulse (single non-zero sample)."""
    signal = np.zeros(n, dtype=np.float64)
    signal[position] = amplitude
    return signal


def generate_dc(n: int, level: float = 1.0) -> np.ndarray:
    """Generate a DC (constant) signal."""
    return np.full(n, level, dtype=np.float64)


def generate_nyquist(n: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate alternating +1/-1 (Nyquist frequency signal)."""
    signal = np.zeros(n, dtype=np.float64)
    for i in range(n):
        signal[i] = amplitude * ((-1) ** i)
    return signal


# =============================================================================
# FFT helpers
# =============================================================================


def compute_fft(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT and return (real, imag) arrays."""
    X = np.fft.fft(signal.astype(np.float64))
    return X.real.astype(np.float64), X.imag.astype(np.float64)


def compute_magnitude(fft_re: np.ndarray, fft_im: np.ndarray) -> np.ndarray:
    """Compute magnitude from real/imag components."""
    return np.sqrt(fft_re**2 + fft_im**2)


def compute_phase(fft_re: np.ndarray, fft_im: np.ndarray) -> np.ndarray:
    """Compute phase from real/imag components."""
    return np.arctan2(fft_im, fft_re)


def find_peak_bin(magnitude: np.ndarray, exclude_dc: bool = True) -> int:
    """Find the bin with maximum magnitude."""
    if exclude_dc and len(magnitude) > 1:
        return int(np.argmax(magnitude[1:])) + 1
    return int(np.argmax(magnitude))


# =============================================================================
# Test case builders
# =============================================================================


@dataclass
class SignalCase:
    name: str
    kind: str
    signal: np.ndarray
    sample_rate: float
    params: dict[str, Any]


def build_pure_sine_cases(sample_rate: float, n: int) -> list[SignalCase]:
    """Build test cases for pure sine waves."""
    cases: list[SignalCase] = []

    # Bin-centered frequencies (exact integer bins)
    for k in [4, 8, 16, 32, 64]:
        if k >= n // 2:
            continue
        freq_hz = k * sample_rate / n
        for amplitude in [0.5, 1.0, 2.0]:
            signal = generate_sine(freq_hz, amplitude, 0.0, sample_rate, n)
            cases.append(
                SignalCase(
                    name=f"sine_bin{k}_amp{amplitude}",
                    kind="pure_sine_bin_centered",
                    signal=signal,
                    sample_rate=sample_rate,
                    params={
                        "frequency_hz": freq_hz,
                        "amplitude": amplitude,
                        "phase_rad": 0.0,
                        "bin_index": k,
                    },
                )
            )

    # Non-bin-centered frequencies (will have spectral leakage)
    for freq_hz in [440.0, 1000.0, 2500.0]:
        if freq_hz >= sample_rate / 2:
            continue
        signal = generate_sine(freq_hz, 1.0, 0.0, sample_rate, n)
        expected_bin = round(freq_hz * n / sample_rate)
        cases.append(
            SignalCase(
                name=f"sine_{int(freq_hz)}hz",
                kind="pure_sine_non_centered",
                signal=signal,
                sample_rate=sample_rate,
                params={
                    "frequency_hz": freq_hz,
                    "amplitude": 1.0,
                    "phase_rad": 0.0,
                    "expected_bin": expected_bin,
                },
            )
        )

    # Phase variations (bin-centered)
    k = 8
    freq_hz = k * sample_rate / n
    for phase_deg in [0, 45, 90, 180, 270]:
        phase_rad = np.deg2rad(phase_deg)
        signal = generate_sine(freq_hz, 1.0, phase_rad, sample_rate, n)
        cases.append(
            SignalCase(
                name=f"sine_bin{k}_phase{phase_deg}deg",
                kind="pure_sine_phase",
                signal=signal,
                sample_rate=sample_rate,
                params={
                    "frequency_hz": freq_hz,
                    "amplitude": 1.0,
                    "phase_rad": phase_rad,
                    "phase_deg": phase_deg,
                    "bin_index": k,
                },
            )
        )

    return cases


def build_cosine_cases(sample_rate: float, n: int) -> list[SignalCase]:
    """Build test cases for cosine waves (phase reference)."""
    cases: list[SignalCase] = []

    k = 8
    freq_hz = k * sample_rate / n

    # Cosine at same frequency as sine for phase comparison
    signal = generate_cosine(freq_hz, 1.0, 0.0, sample_rate, n)
    cases.append(
        SignalCase(
            name=f"cosine_bin{k}",
            kind="cosine",
            signal=signal,
            sample_rate=sample_rate,
            params={
                "frequency_hz": freq_hz,
                "amplitude": 1.0,
                "phase_rad": 0.0,
                "bin_index": k,
            },
        )
    )

    return cases


def build_multi_tone_cases(sample_rate: float, n: int) -> list[SignalCase]:
    """Build test cases for multi-tone signals."""
    cases: list[SignalCase] = []

    # Two tones
    k1, k2 = 8, 24
    f1 = k1 * sample_rate / n
    f2 = k2 * sample_rate / n
    signal = generate_multi_tone([f1, f2], [1.0, 0.5], [0.0, 0.0], sample_rate, n)
    cases.append(
        SignalCase(
            name=f"two_tone_bin{k1}_bin{k2}",
            kind="multi_tone",
            signal=signal,
            sample_rate=sample_rate,
            params={
                "frequencies_hz": [f1, f2],
                "amplitudes": [1.0, 0.5],
                "phases_rad": [0.0, 0.0],
                "bin_indices": [k1, k2],
            },
        )
    )

    # Three tones
    k1, k2, k3 = 4, 16, 48
    f1 = k1 * sample_rate / n
    f2 = k2 * sample_rate / n
    f3 = k3 * sample_rate / n
    signal = generate_multi_tone(
        [f1, f2, f3], [0.8, 1.0, 0.3], [0.0, 0.0, 0.0], sample_rate, n
    )
    cases.append(
        SignalCase(
            name=f"three_tone_bin{k1}_bin{k2}_bin{k3}",
            kind="multi_tone",
            signal=signal,
            sample_rate=sample_rate,
            params={
                "frequencies_hz": [f1, f2, f3],
                "amplitudes": [0.8, 1.0, 0.3],
                "phases_rad": [0.0, 0.0, 0.0],
                "bin_indices": [k1, k2, k3],
            },
        )
    )

    return cases


def build_chirp_cases(sample_rate: float, n: int) -> list[SignalCase]:
    """Build test cases for chirp signals."""
    cases: list[SignalCase] = []

    # Low to mid chirp
    f0, f1 = 100.0, 2000.0
    signal = generate_chirp(f0, f1, sample_rate, n)
    cases.append(
        SignalCase(
            name=f"chirp_{int(f0)}hz_to_{int(f1)}hz",
            kind="chirp",
            signal=signal,
            sample_rate=sample_rate,
            params={
                "f0_hz": f0,
                "f1_hz": f1,
                "amplitude": 1.0,
            },
        )
    )

    return cases


def build_special_cases(sample_rate: float, n: int) -> list[SignalCase]:
    """Build special test cases (impulse, DC, Nyquist, zeros)."""
    cases: list[SignalCase] = []

    # Impulse at position 0
    signal = generate_impulse(n, position=0, amplitude=1.0)
    cases.append(
        SignalCase(
            name="impulse_pos0",
            kind="impulse",
            signal=signal,
            sample_rate=sample_rate,
            params={"position": 0, "amplitude": 1.0},
        )
    )

    # Impulse at middle
    mid = n // 2
    signal = generate_impulse(n, position=mid, amplitude=1.0)
    cases.append(
        SignalCase(
            name=f"impulse_pos{mid}",
            kind="impulse",
            signal=signal,
            sample_rate=sample_rate,
            params={"position": mid, "amplitude": 1.0},
        )
    )

    # DC signal
    signal = generate_dc(n, level=1.0)
    cases.append(
        SignalCase(
            name="dc_level1",
            kind="dc",
            signal=signal,
            sample_rate=sample_rate,
            params={"level": 1.0},
        )
    )

    # DC + sine (offset)
    k = 8
    freq_hz = k * sample_rate / n
    dc_level = 0.5
    sine = generate_sine(freq_hz, 1.0, 0.0, sample_rate, n)
    signal = generate_dc(n, level=dc_level) + sine
    cases.append(
        SignalCase(
            name=f"dc_plus_sine_bin{k}",
            kind="dc_plus_sine",
            signal=signal,
            sample_rate=sample_rate,
            params={
                "dc_level": dc_level,
                "sine_frequency_hz": freq_hz,
                "sine_amplitude": 1.0,
                "sine_bin": k,
            },
        )
    )

    # Nyquist signal (alternating +1/-1)
    signal = generate_nyquist(n, amplitude=1.0)
    cases.append(
        SignalCase(
            name="nyquist",
            kind="nyquist",
            signal=signal,
            sample_rate=sample_rate,
            params={"amplitude": 1.0},
        )
    )

    # All zeros
    signal = np.zeros(n, dtype=np.float64)
    cases.append(
        SignalCase(
            name="zeros",
            kind="zeros",
            signal=signal,
            sample_rate=sample_rate,
            params={},
        )
    )

    # Very small values
    signal = generate_sine(8 * sample_rate / n, 1e-12, 0.0, sample_rate, n)
    cases.append(
        SignalCase(
            name="tiny_amplitude",
            kind="tiny",
            signal=signal,
            sample_rate=sample_rate,
            params={"amplitude": 1e-12},
        )
    )

    # Large values
    signal = generate_sine(8 * sample_rate / n, 1e6, 0.0, sample_rate, n)
    cases.append(
        SignalCase(
            name="large_amplitude",
            kind="large",
            signal=signal,
            sample_rate=sample_rate,
            params={"amplitude": 1e6},
        )
    )

    return cases


def build_window_dsp_cases(sizes: list[int]) -> list[dict[str, Any]]:
    """Build window DSP property test cases."""
    window_types = ["rect", "hann", "hamming", "blackman"]
    cases: list[dict[str, Any]] = []

    for n in sizes:
        for wtype in window_types:
            if wtype == "rect":
                w = np.ones(n, dtype=np.float64)
            elif wtype == "hann":
                w = sp_windows.hann(n, sym=True).astype(np.float64)
            elif wtype == "hamming":
                w = sp_windows.hamming(n, sym=True).astype(np.float64)
            elif wtype == "blackman":
                w = sp_windows.blackman(n, sym=True).astype(np.float64)
            else:
                continue

            # Coherent gain: sum(w) / N
            coherent_gain = float(np.sum(w) / n)

            # ENBW: N * sum(w^2) / sum(w)^2
            enbw = float(n * np.sum(w**2) / np.sum(w) ** 2)

            cases.append(
                {
                    "type": wtype,
                    "n": n,
                    "values": to_float_list(w),
                    "coherentGain": coherent_gain,
                    "enbw": enbw,
                }
            )

    return cases


# =============================================================================
# Output generation
# =============================================================================


def case_to_dict(case: SignalCase) -> dict[str, Any]:
    """Convert a SignalCase to a dictionary for JSON output."""
    fft_re, fft_im = compute_fft(case.signal)
    magnitude = compute_magnitude(fft_re, fft_im)
    phase = compute_phase(fft_re, fft_im)
    peak_bin = find_peak_bin(magnitude, exclude_dc=(case.kind != "dc"))

    return {
        "name": case.name,
        "kind": case.kind,
        "n": len(case.signal),
        "sampleRate": case.sample_rate,
        "signal": to_float_list(case.signal),
        "fftRe": to_float_list(fft_re),
        "fftIm": to_float_list(fft_im),
        "magnitude": to_float_list(magnitude),
        "phase": to_float_list(phase),
        "peakBin": peak_bin,
        "peakMagnitude": float(magnitude[peak_bin]),
        "peakPhase": float(phase[peak_bin]),
        "params": case.params,
    }


def generate_all_references(
    out_dir: str,
    sample_rate: float = 48000.0,
    n: int = 1024,
    window_sizes: list[int] | None = None,
) -> None:
    """Generate all reference files."""
    os.makedirs(out_dir, exist_ok=True)

    if window_sizes is None:
        window_sizes = [64, 256, 1024, 2048]

    meta = generator_meta()

    # Pure sine cases
    sine_cases = build_pure_sine_cases(sample_rate, n)
    sine_output = {
        **meta,
        "description": "Pure sine wave test cases",
        "n": n,
        "sampleRate": sample_rate,
        "cases": [case_to_dict(c) for c in sine_cases],
    }
    with open(os.path.join(out_dir, "pure_sine.json"), "w") as f:
        json.dump(sine_output, f, indent=2)
    print(f"Wrote {len(sine_cases)} cases to pure_sine.json")

    # Cosine cases (for phase reference)
    cosine_cases = build_cosine_cases(sample_rate, n)
    cosine_output = {
        **meta,
        "description": "Cosine wave test cases for phase reference",
        "n": n,
        "sampleRate": sample_rate,
        "cases": [case_to_dict(c) for c in cosine_cases],
    }
    with open(os.path.join(out_dir, "cosine.json"), "w") as f:
        json.dump(cosine_output, f, indent=2)
    print(f"Wrote {len(cosine_cases)} cases to cosine.json")

    # Multi-tone cases
    multi_cases = build_multi_tone_cases(sample_rate, n)
    multi_output = {
        **meta,
        "description": "Multi-tone signal test cases",
        "n": n,
        "sampleRate": sample_rate,
        "cases": [case_to_dict(c) for c in multi_cases],
    }
    with open(os.path.join(out_dir, "multi_tone.json"), "w") as f:
        json.dump(multi_output, f, indent=2)
    print(f"Wrote {len(multi_cases)} cases to multi_tone.json")

    # Chirp cases
    chirp_cases = build_chirp_cases(sample_rate, n)
    chirp_output = {
        **meta,
        "description": "Chirp signal test cases",
        "n": n,
        "sampleRate": sample_rate,
        "cases": [case_to_dict(c) for c in chirp_cases],
    }
    with open(os.path.join(out_dir, "chirp.json"), "w") as f:
        json.dump(chirp_output, f, indent=2)
    print(f"Wrote {len(chirp_cases)} cases to chirp.json")

    # Special cases (impulse, DC, Nyquist, zeros, edge values)
    special_cases = build_special_cases(sample_rate, n)
    special_output = {
        **meta,
        "description": "Special signal test cases (impulse, DC, Nyquist, edge values)",
        "n": n,
        "sampleRate": sample_rate,
        "cases": [case_to_dict(c) for c in special_cases],
    }
    with open(os.path.join(out_dir, "special.json"), "w") as f:
        json.dump(special_output, f, indent=2)
    print(f"Wrote {len(special_cases)} cases to special.json")

    # Window DSP properties
    window_cases = build_window_dsp_cases(window_sizes)
    window_output = {
        **meta,
        "description": "Window function DSP properties",
        "cases": window_cases,
    }
    with open(os.path.join(out_dir, "windows_dsp.json"), "w") as f:
        json.dump(window_output, f, indent=2)
    print(f"Wrote {len(window_cases)} cases to windows_dsp.json")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate real-life test references")
    p.add_argument(
        "--out",
        default="test/reallife/references/",
        help="Output directory for reference JSON files",
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=48000.0,
        help="Sample rate for test signals (Hz)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=1024,
        help="FFT size for test signals",
    )
    p.add_argument(
        "--window-sizes",
        default="64,256,1024,2048",
        help="Comma-separated window sizes for DSP property tests",
    )
    args = p.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]

    generate_all_references(
        out_dir=args.out,
        sample_rate=args.sample_rate,
        n=args.n,
        window_sizes=window_sizes,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
