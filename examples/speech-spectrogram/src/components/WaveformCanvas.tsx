import { useRef, useEffect } from "react";

type Props = {
  /** Mono audio samples */
  samples: Float32Array | Float64Array;
  sampleRate: number;
  /** Optional label drawn in the top-left */
  label?: string;
  width?: number;
  height?: number;
  color?: string;
};

/**
 * Draws a time-domain waveform using raw Canvas2D.
 */
export function WaveformCanvas({
  samples,
  sampleRate,
  label,
  width = 800,
  height = 160,
  color = "#4fc3f7",
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

    // Background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    // Zero line
    const midY = height / 2;
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    if (samples.length === 0) return;

    // Find peak for scaling
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]!);
      if (abs > peak) peak = abs;
    }
    if (peak === 0) peak = 1;

    // Draw waveform
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();

    const step = samples.length / width;
    for (let px = 0; px < width; px++) {
      const idx = Math.floor(px * step);
      const val = samples[idx]! / peak;
      const y = midY - val * (midY - 4);
      if (px === 0) ctx.moveTo(px, y);
      else ctx.lineTo(px, y);
    }
    ctx.stroke();

    // Time axis labels
    ctx.fillStyle = "#888";
    ctx.font = "11px monospace";
    const duration = samples.length / sampleRate;
    for (let t = 0; t <= duration; t += Math.max(0.5, Math.floor(duration / 6))) {
      const px = (t / duration) * width;
      ctx.fillText(`${t.toFixed(1)}s`, px + 2, height - 4);
    }

    // Label
    if (label) {
      ctx.fillStyle = "#ccc";
      ctx.font = "bold 12px monospace";
      ctx.fillText(label, 6, 16);
    }
  }, [samples, sampleRate, label, width, height, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, display: "block", borderRadius: 4 }}
    />
  );
}
