import { useEffect, useRef, useState } from "react";

import { magnitudeToDb } from "pragma-dsp/analysis";
import type { StftResult } from "pragma-dsp/xform/stft";

import type { FormantResult } from "../speech-analysis";

type SpectrogramCanvasProps = {
  readonly stft: StftResult;
  readonly formants?: ReadonlyArray<FormantResult>;
  readonly pitchTrack?: ReadonlyArray<number | null>;
  readonly width?: number;
  readonly height?: number;
  readonly maxFreqDisplay?: number;
  readonly className?: string;
};

type Rgb = readonly [number, number, number];

const paletteStops: ReadonlyArray<readonly [number, Rgb]> = [
  [0.0, [4, 2, 14]],
  [0.1, [18, 9, 54]],
  [0.22, [48, 11, 101]],
  [0.38, [94, 18, 120]],
  [0.52, [148, 35, 106]],
  [0.68, [205, 52, 70]],
  [0.82, [243, 108, 33]],
  [0.92, [251, 181, 38]],
  [1.0, [252, 248, 164]],
];

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const clamp01 = (value: number) => clamp(value, 0, 1);

const lerp = (start: number, end: number, t: number) => start + (end - start) * t;

const inferno = (value: number): Rgb => {
  const normalized = clamp01(value);

  for (let index = 1; index < paletteStops.length; index++) {
    const left = paletteStops[index - 1];
    const right = paletteStops[index];
    if (!left || !right || normalized > right[0]) {
      continue;
    }

    const localT = (normalized - left[0]) / Math.max(1e-6, right[0] - left[0]);
    return [
      Math.round(lerp(left[1][0], right[1][0], localT)),
      Math.round(lerp(left[1][1], right[1][1], localT)),
      Math.round(lerp(left[1][2], right[1][2], localT)),
    ];
  }

  return paletteStops[paletteStops.length - 1]?.[1] ?? [252, 248, 164];
};

const collectQuantile = (values: ArrayLike<number>, ratio: number) => {
  const sorted = Array.from(values).sort((left, right) => left - right);
  if (sorted.length === 0) {
    return 0;
  }

  const index = Math.round((sorted.length - 1) * clamp01(ratio));
  return sorted[index] ?? 0;
};

const sampleDisplayValues = (
  frames: ReadonlyArray<Float64Array>,
  bins: number,
  frequencies: Float64Array,
  maxSamples = 42_000,
) => {
  const total = frames.length * bins;
  const step = Math.max(1, Math.ceil(total / maxSamples));
  const values: number[] = [];
  let cursor = 0;

  for (let frameIndex = 0; frameIndex < frames.length; frameIndex++) {
    const frame = frames[frameIndex] ?? new Float64Array(0);
    for (let binIndex = 0; binIndex < bins; binIndex++) {
      if (cursor % step === 0) {
        const frequency = frequencies[binIndex] ?? 0;
        const spectralTiltLift = 6.5 * Math.log2(1 + frequency / 220);
        values.push((frame[binIndex] ?? -110) + spectralTiltLift);
      }
      cursor += 1;
    }
  }

  return values;
};

const formatFrequencyLabel = (frequency: number) => {
  if (frequency === 0) {
    return "0 Hz";
  }

  if (frequency >= 1000) {
    const value = frequency / 1000;
    return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} kHz`;
  }

  return `${Math.round(frequency)} Hz`;
};

const formatTimeLabel = (seconds: number) => `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;

const formatRelativeDbLabel = (value: number, ceiling: number) =>
  `${Math.round(value - ceiling)} dB`;

const estimateDisplayBins = (frequencies: Float64Array, maxFrequency: number) => {
  for (let index = 0; index < frequencies.length; index++) {
    if ((frequencies[index] ?? 0) > maxFrequency) {
      return Math.max(1, index);
    }
  }

  return Math.max(1, frequencies.length);
};

const createNiceTimeTicks = (duration: number) => {
  if (duration <= 0) {
    return [0];
  }

  const targetTickCount = 6;
  const roughStep = duration / targetTickCount;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(roughStep, 1e-3)));
  const normalized = roughStep / magnitude;
  const step =
    normalized <= 1
      ? 1 * magnitude
      : normalized <= 2
        ? 2 * magnitude
        : normalized <= 5
          ? 5 * magnitude
          : 10 * magnitude;

  const ticks: number[] = [];
  for (let value = 0; value <= duration + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
};

export function SpectrogramCanvas({
  stft,
  formants,
  pitchTrack,
  width,
  height = 320,
  maxFreqDisplay,
  className,
}: SpectrogramCanvasProps) {
  "use no memo";

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [measuredWidth, setMeasuredWidth] = useState(width ?? 0);

  useEffect(() => {
    if (width != null) {
      setMeasuredWidth(width);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const updateWidth = () => {
      const nextWidth = Math.floor(canvas.getBoundingClientRect().width);
      if (nextWidth > 0) {
        setMeasuredWidth(nextWidth);
      }
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(canvas);

    return () => observer.disconnect();
  }, [width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || measuredWidth <= 0) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(measuredWidth * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.scale(dpr, dpr);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";

    const { frames, frequencies, sampleRate, times } = stft;

    context.fillStyle = "#05070b";
    context.fillRect(0, 0, measuredWidth, height);

    if (frames.length === 0 || frequencies.length === 0) {
      return;
    }

    const displayMax = Math.min(maxFreqDisplay ?? sampleRate / 2, sampleRate / 2);
    const displayBins = estimateDisplayBins(frequencies, displayMax);

    const margins = {
      top: 14,
      right: 88,
      bottom: 30,
      left: 60,
    };
    const plotLeft = margins.left;
    const plotTop = margins.top;
    const plotWidth = Math.max(1, measuredWidth - margins.left - margins.right);
    const plotHeight = Math.max(1, height - margins.top - margins.bottom);
    const plotRight = plotLeft + plotWidth;
    const plotBottom = plotTop + plotHeight;
    const lastTime = times[times.length - 1] ?? 0;

    const dbFrames = frames.map((frame) =>
      magnitudeToDb(frame.magnitudes, { floorDb: -110 }),
    );
    const displaySamples = sampleDisplayValues(
      dbFrames,
      displayBins,
      frequencies,
    );
    const displayCeil = Math.max(-22, collectQuantile(displaySamples, 0.996));
    const displayFloor = Math.min(displayCeil - 26, collectQuantile(displaySamples, 0.06));
    const displayRange = Math.max(1, displayCeil - displayFloor);

    const frequencyToY = (frequency: number) => {
      const normalized = clamp01(frequency / displayMax);
      return plotBottom - Math.pow(normalized, 0.58) * plotHeight;
    };

    const rowToFrequency = (row: number) => {
      const ratio = 1 - row / Math.max(1, plotHeight - 1);
      return Math.pow(ratio, 1 / 0.58) * displayMax;
    };

    const binLookup = new Uint16Array(plotHeight);
    for (let row = 0; row < plotHeight; row++) {
      const frequency = rowToFrequency(row);
      let low = 0;
      let high = displayBins - 1;

      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        const midFrequency = frequencies[mid] ?? 0;
        if (midFrequency < frequency) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }

      const upperIndex = clamp(low, 0, displayBins - 1);
      const lowerIndex = clamp(upperIndex - 1, 0, displayBins - 1);
      const currentFrequency = frequencies[lowerIndex] ?? 0;
      const nextFrequency = frequencies[upperIndex] ?? currentFrequency;
      binLookup[row] =
        Math.abs(frequency - nextFrequency) < Math.abs(frequency - currentFrequency)
          ? upperIndex
          : lowerIndex;
    }

    const heatmap = context.createImageData(frames.length, plotHeight);
    for (let frameIndex = 0; frameIndex < frames.length; frameIndex++) {
      const dbFrame = dbFrames[frameIndex] ?? new Float64Array(0);
      for (let row = 0; row < plotHeight; row++) {
        const binIndex = binLookup[row] ?? 0;
        const frequency = frequencies[binIndex] ?? 0;
        const rawDb = dbFrame[binIndex] ?? -110;
        const spectralTiltLift = 6.5 * Math.log2(1 + frequency / 220);
        const weightedDb = rawDb + spectralTiltLift;
        const normalized = clamp01((weightedDb - displayFloor) / displayRange);
        const contourBoost = Math.pow(normalized, 0.94);
        const [red, green, blue] = inferno(contourBoost);
        const pixelIndex = (row * frames.length + frameIndex) * 4;
        heatmap.data[pixelIndex] = red;
        heatmap.data[pixelIndex + 1] = green;
        heatmap.data[pixelIndex + 2] = blue;
        heatmap.data[pixelIndex + 3] = 255;
      }
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = frames.length;
    offscreen.height = plotHeight;
    const offscreenContext = offscreen.getContext("2d");
    offscreenContext?.putImageData(heatmap, 0, 0);

    context.fillStyle = "#05070b";
    context.fillRect(plotLeft, plotTop, plotWidth, plotHeight);
    context.drawImage(offscreen, plotLeft, plotTop, plotWidth, plotHeight);

    const gridFrequencies = [250, 500, 1000, 2000, 4000, 8000].filter(
      (frequency) => frequency <= displayMax,
    );
    const timeTicks = createNiceTimeTicks(lastTime);

    context.strokeStyle = "rgba(255,255,255,0.08)";
    context.lineWidth = 1;
    for (const frequency of gridFrequencies) {
      const y = frequencyToY(frequency);
      context.beginPath();
      context.moveTo(plotLeft, y + 0.5);
      context.lineTo(plotRight, y + 0.5);
      context.stroke();
    }
    for (const time of timeTicks) {
      const x = plotLeft + (time / Math.max(lastTime, 1e-6)) * plotWidth;
      context.beginPath();
      context.moveTo(x + 0.5, plotTop);
      context.lineTo(x + 0.5, plotBottom);
      context.stroke();
    }

    context.strokeStyle = "rgba(255,255,255,0.18)";
    context.strokeRect(plotLeft + 0.5, plotTop + 0.5, plotWidth - 1, plotHeight - 1);

    context.save();
    context.beginPath();
    context.rect(plotLeft, plotTop, plotWidth, plotHeight);
    context.clip();

    if (pitchTrack && pitchTrack.length > 0) {
      context.strokeStyle = "rgba(255, 239, 173, 0.95)";
      context.lineWidth = 1.7;
      context.shadowColor = "rgba(255, 214, 107, 0.18)";
      context.shadowBlur = 4;
      context.beginPath();
      let open = false;

      for (let frameIndex = 0; frameIndex < Math.min(frames.length, pitchTrack.length); frameIndex++) {
        const f0 = pitchTrack[frameIndex];
        if (f0 == null || f0 <= 0 || f0 > displayMax) {
          open = false;
          continue;
        }

        const x = plotLeft + (frameIndex / Math.max(1, frames.length - 1)) * plotWidth;
        const y = frequencyToY(f0);
        if (!open) {
          context.moveTo(x, y);
          open = true;
        } else {
          context.lineTo(x, y);
        }
      }

      context.stroke();
      context.shadowBlur = 0;
    }

    if (formants) {
      const colors = ["#fff5c6", "#7de2ff", "#f29cff", "#96ffaf"];
      for (let frameIndex = 0; frameIndex < Math.min(formants.length, frames.length); frameIndex++) {
        const frame = formants[frameIndex];
        if (!frame) {
          continue;
        }

        const x = plotLeft + (frameIndex / Math.max(1, frames.length - 1)) * plotWidth;
        for (let formantIndex = 0; formantIndex < frame.formants.length; formantIndex++) {
          const frequency = frame.formants[formantIndex];
          if (frequency == null || frequency <= 0 || frequency > displayMax) {
            continue;
          }

          context.fillStyle = colors[formantIndex % colors.length] ?? "#ffffff";
          context.beginPath();
          context.arc(x, frequencyToY(frequency), 2.1, 0, Math.PI * 2);
          context.fill();
        }
      }
    }
    context.restore();

    context.fillStyle = "rgba(247, 245, 239, 0.92)";
    context.font = '11px "Geist Variable", ui-sans-serif, sans-serif';
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (const frequency of [0, ...gridFrequencies]) {
      context.fillText(formatFrequencyLabel(frequency), plotLeft - 8, frequencyToY(frequency));
    }

    context.textAlign = "center";
    context.textBaseline = "top";
    for (const time of timeTicks) {
      const x = plotLeft + (time / Math.max(lastTime, 1e-6)) * plotWidth;
      context.fillText(formatTimeLabel(time), x, plotBottom + 6);
    }

    const barWidth = 12;
    const barLeft = plotRight + 26;
    const barTop = plotTop;
    const barHeight = plotHeight;
    const gradient = context.createLinearGradient(0, barTop + barHeight, 0, barTop);
    for (let index = 0; index <= 100; index++) {
      const t = index / 100;
      const [red, green, blue] = inferno(t);
      gradient.addColorStop(t, `rgb(${red}, ${green}, ${blue})`);
    }
    context.fillStyle = gradient;
    context.fillRect(barLeft, barTop, barWidth, barHeight);
    context.strokeStyle = "rgba(255,255,255,0.2)";
    context.strokeRect(barLeft + 0.5, barTop + 0.5, barWidth - 1, barHeight - 1);

    const colorTicks = Array.from({ length: 5 }, (_, index) =>
      lerp(displayFloor, displayCeil, index / 4),
    );
    context.fillStyle = "rgba(247, 245, 239, 0.75)";
    context.textAlign = "left";
    context.textBaseline = "middle";
    for (const tick of colorTicks) {
      const y = barTop + (1 - (tick - displayFloor) / displayRange) * barHeight;
      context.fillText(formatRelativeDbLabel(tick, displayCeil), barLeft + barWidth + 8, y);
    }
  }, [formants, height, maxFreqDisplay, measuredWidth, pitchTrack, stft]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: width ?? "100%", height, borderRadius: 8 }}
    />
  );
}
