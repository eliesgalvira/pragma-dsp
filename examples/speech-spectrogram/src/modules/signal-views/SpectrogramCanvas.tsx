import { useEffect, useRef } from "react";

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
};

const viridis = (value: number): [number, number, number] => {
  const clamped = Math.max(0, Math.min(1, value));
  const red = Math.round(255 * Math.min(1, Math.max(0, -1.4 * clamped * clamped + 2.2 * clamped + 0.15)));
  const green = Math.round(255 * Math.min(1, Math.max(0, -0.6 * clamped * clamped + 1.2 * clamped + 0.1)));
  const blue = Math.round(255 * Math.min(1, Math.max(0, 0.8 - 1.5 * clamped + 0.7 * clamped * clamped)));
  return [red, green, blue];
};

export function SpectrogramCanvas({
  stft,
  formants,
  pitchTrack,
  width = 820,
  height = 320,
  maxFreqDisplay,
}: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.scale(dpr, dpr);

    const { frames, frequencies, sampleRate } = stft;
    const displayMax = Math.min(maxFreqDisplay ?? sampleRate / 2, sampleRate / 2);

    context.fillStyle = "#091217";
    context.fillRect(0, 0, width, height);

    if (frames.length === 0 || frequencies.length === 0) {
      return;
    }

    let displayBins = frequencies.length;
    for (let index = 0; index < frequencies.length; index++) {
      if ((frequencies[index] ?? 0) > displayMax) {
        displayBins = index + 1;
        break;
      }
    }

    const dbFrames = frames.map((frame) =>
      magnitudeToDb(frame.magnitudes, { floorDb: -80 }),
    );

    let maxDb = -80;
    for (const frame of dbFrames) {
      for (let index = 0; index < displayBins; index++) {
        maxDb = Math.max(maxDb, frame[index] ?? -80);
      }
    }

    const dynamicRange = Math.max(1, maxDb + 80);
    const heatmap = context.createImageData(frames.length, displayBins);

    for (let frameIndex = 0; frameIndex < frames.length; frameIndex++) {
      const dbFrame = dbFrames[frameIndex] ?? new Float64Array(0);
      for (let binIndex = 0; binIndex < displayBins; binIndex++) {
        const normalized = ((dbFrame[binIndex] ?? -80) + 80) / dynamicRange;
        const [red, green, blue] = viridis(normalized);
        const flippedY = displayBins - binIndex - 1;
        const pixelIndex = (flippedY * frames.length + frameIndex) * 4;
        heatmap.data[pixelIndex] = red;
        heatmap.data[pixelIndex + 1] = green;
        heatmap.data[pixelIndex + 2] = blue;
        heatmap.data[pixelIndex + 3] = 255;
      }
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = frames.length;
    offscreen.height = displayBins;
    offscreen.getContext("2d")?.putImageData(heatmap, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(offscreen, 0, 0, width, height);

    if (pitchTrack && pitchTrack.length > 0) {
      context.strokeStyle = "#ff9d5c";
      context.lineWidth = 2;
      context.beginPath();
      let drawing = false;

      for (let index = 0; index < frames.length; index++) {
        const f0 = pitchTrack[index];
        if (f0 == null || f0 <= 0 || f0 > displayMax) {
          drawing = false;
          continue;
        }

        const x = (index / frames.length) * width + width / frames.length / 2;
        const y = height - (f0 / displayMax) * height;
        if (!drawing) {
          context.moveTo(x, y);
          drawing = true;
        } else {
          context.lineTo(x, y);
        }
      }

      context.stroke();
    }

    if (formants) {
      const colors = ["#ffd166", "#54d4c4", "#7aa2ff", "#f776c6"];
      for (let frameIndex = 0; frameIndex < Math.min(formants.length, frames.length); frameIndex++) {
        const frame = formants[frameIndex];
        if (!frame) {
          continue;
        }

        const x = (frameIndex / frames.length) * width + width / frames.length / 2;
        for (let formantIndex = 0; formantIndex < frame.formants.length; formantIndex++) {
          const frequency = frame.formants[formantIndex];
          if (frequency == null || frequency > displayMax) {
            continue;
          }

          context.fillStyle = colors[formantIndex % colors.length] ?? "#fff";
          context.beginPath();
          context.arc(x, height - (frequency / displayMax) * height, 2.2, 0, Math.PI * 2);
          context.fill();
        }
      }
    }

    context.fillStyle = "rgba(247, 245, 239, 0.65)";
    context.font = "11px JetBrains Mono";
    for (let stepIndex = 0; stepIndex <= 5; stepIndex++) {
      const frequency = (displayMax / 5) * stepIndex;
      const y = height - (height / 5) * stepIndex;
      context.fillText(`${Math.round(frequency)} Hz`, 8, Math.max(14, y - 4));
    }

    const lastTime = stft.times[stft.times.length - 1] ?? 0;
    if (lastTime > 0) {
      for (let stepIndex = 0; stepIndex <= 5; stepIndex++) {
        const time = (lastTime / 5) * stepIndex;
        const x = (width / 5) * stepIndex;
        context.fillText(`${time.toFixed(1)}s`, Math.min(width - 40, x + 4), height - 8);
      }
    }
  }, [formants, height, maxFreqDisplay, pitchTrack, stft, width]);

  return <canvas ref={canvasRef} style={{ width, height, borderRadius: 16 }} />;
}
