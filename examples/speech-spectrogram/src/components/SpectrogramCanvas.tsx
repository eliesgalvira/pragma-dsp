import { useRef, useEffect } from "react";
import type { StftResult } from "pragma-dsp/xform/stft";
import { magnitudeToDb } from "pragma-dsp/analysis";
import type { FormantResult } from "../dsp/pitch";

type Props = {
  stft: StftResult;
  /** Optional formant markers per frame (parallel array to stft.frames) */
  formants?: FormantResult[];
  /** F0 markers per frame */
  pitchTrack?: (number | null)[];
  width?: number;
  height?: number;
  /** Max frequency to display (Hz). Defaults to sampleRate/2. */
  maxFreqDisplay?: number;
};

/** Viridis-ish colormap (5-stop linear interpolation). */
function viridis(t: number): [number, number, number] {
  // t in [0,1]
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, -1.4 * c * c + 2.2 * c + 0.15)));
  const g = Math.round(255 * Math.min(1, Math.max(0, -0.6 * c * c + 1.2 * c + 0.1)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.8 - 1.5 * c + 0.7 * c * c)));
  return [r, g, b];
}

/**
 * Draws a spectrogram heatmap using raw Canvas2D.
 * X-axis = time, Y-axis = frequency (linear, low at bottom).
 */
export function SpectrogramCanvas({
  stft,
  formants,
  pitchTrack,
  width = 800,
  height = 300,
  maxFreqDisplay,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const { frames, frequencies, sampleRate } = stft;
    const nyquist = sampleRate / 2;
    const displayMax = maxFreqDisplay ?? nyquist;
    const numFrames = frames.length;
    const numBins = frequencies.length;

    // Determine which bins to display
    let maxBin = numBins - 1;
    for (let i = 0; i < numBins; i++) {
      if (frequencies[i]! > displayMax) {
        maxBin = i;
        break;
      }
    }
    const displayBins = maxBin + 1;

    // Convert all frames to dB
    const dbFrames = frames.map((f) =>
      magnitudeToDb(f.magnitudes, { floorDb: -80 })
    );

    // Global min/max for color scaling
    let globalMin = 0, globalMax = -Infinity;
    for (const db of dbFrames) {
      for (let i = 0; i < displayBins; i++) {
        const v = db[i]!;
        if (v > globalMax) globalMax = v;
        if (v < globalMin) globalMin = v;
      }
    }
    globalMin = -80;
    const range = globalMax - globalMin || 1;

    // Background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    // Draw heatmap
    const colW = width / numFrames;
    const rowH = height / displayBins;

    const imageData = ctx.createImageData(numFrames, displayBins);
    const pixels = imageData.data;

    for (let f = 0; f < numFrames; f++) {
      const db = dbFrames[f]!;
      for (let b = 0; b < displayBins; b++) {
        const t = (db[b]! - globalMin) / range;
        const [r, g, bl] = viridis(t);
        // Flip Y: bin 0 (DC) at bottom
        const y = displayBins - 1 - b;
        const idx = (y * numFrames + f) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = bl;
        pixels[idx + 3] = 255;
      }
    }

    // Scale imageData to canvas
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = numFrames;
    tmpCanvas.height = displayBins;
    tmpCanvas.getContext("2d")!.putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmpCanvas, 0, 0, width, height);

    // ── Overlay: pitch track ────
    if (pitchTrack && pitchTrack.length > 0) {
      ctx.strokeStyle = "#ff5722";
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let f = 0; f < numFrames; f++) {
        const f0 = pitchTrack[f];
        if (f0 == null || f0 <= 0 || f0 > displayMax) {
          started = false;
          continue;
        }
        const x = (f / numFrames) * width + colW / 2;
        const y = height - (f0 / displayMax) * height;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // ── Overlay: formant dots ────
    if (formants) {
      const colors = ["#ffd600", "#00e676", "#40c4ff", "#ea80fc"];
      for (let f = 0; f < Math.min(numFrames, formants.length); f++) {
        const fr = formants[f]!;
        const x = (f / numFrames) * width + colW / 2;
        for (let fi = 0; fi < fr.formants.length; fi++) {
          const freq = fr.formants[fi]!;
          if (freq > displayMax) continue;
          const y = height - (freq / displayMax) * height;
          ctx.fillStyle = colors[fi % colors.length]!;
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // ── Axis labels ────
    ctx.fillStyle = "#888";
    ctx.font = "11px monospace";

    // Frequency axis (left)
    const freqSteps = 5;
    for (let i = 0; i <= freqSteps; i++) {
      const freq = (i / freqSteps) * displayMax;
      const y = height - (i / freqSteps) * height;
      ctx.fillText(`${Math.round(freq)}`, 4, y - 2);
    }

    // Time axis (bottom)
    if (stft.times.length > 0) {
      const duration = stft.times[stft.times.length - 1]!;
      for (let t = 0; t <= duration; t += Math.max(0.5, Math.floor(duration / 6))) {
        const x = (t / duration) * width;
        ctx.fillText(`${t.toFixed(1)}s`, x + 2, height - 4);
      }
    }
  }, [stft, formants, pitchTrack, width, height, maxFreqDisplay]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, display: "block", borderRadius: 4 }}
    />
  );
}
