import { useEffect, useRef, useState } from "react";

type WaveformCanvasProps = {
  readonly samples: Float32Array | Float64Array;
  readonly sampleRate: number;
  readonly label?: string;
  readonly width?: number;
  readonly height?: number;
  readonly color?: string;
  readonly amplitudeReference?: number;
  readonly className?: string;
};

export function WaveformCanvas({
  samples,
  sampleRate,
  label,
  width,
  height = 180,
  color = "#54d4c4",
  amplitudeReference,
  className,
}: WaveformCanvasProps) {
  "use no memo";

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [measuredWidth, setMeasuredWidth] = useState(0);
  const renderWidth = width ?? measuredWidth;

  useEffect(() => {
    if (width != null) {
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
    if (!canvas || renderWidth <= 0) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = renderWidth * dpr;
    canvas.height = height * dpr;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.scale(dpr, dpr);

    context.fillStyle = "#091217";
    context.fillRect(0, 0, renderWidth, height);

    context.strokeStyle = "rgba(255, 255, 255, 0.08)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(renderWidth, height / 2);
    context.stroke();

    if (samples.length > 0) {
      let peak = 0;
      for (let index = 0; index < samples.length; index++) {
        peak = Math.max(peak, Math.abs(samples[index] ?? 0));
      }

      const scale = Math.max(amplitudeReference ?? peak, 1e-3);

      const pixels = Math.max(1, Math.floor(renderWidth));
      const samplesPerPixel = Math.max(1, Math.ceil(samples.length / pixels));

      context.strokeStyle = color;
      context.lineWidth = 1;

      for (let pixel = 0; pixel < pixels; pixel++) {
        const start = pixel * samplesPerPixel;
        const end = Math.min(samples.length, start + samplesPerPixel);
        if (start >= end) {
          continue;
        }

        const first = (samples[start] ?? 0) / scale;
        let min = first;
        let max = first;

        for (let index = start + 1; index < end; index++) {
          const value = (samples[index] ?? 0) / scale;
          if (value < min) min = value;
          if (value > max) max = value;
        }

        const y1 = height / 2 - max * (height / 2 - 10);
        const y2 = height / 2 - min * (height / 2 - 10);

        context.beginPath();
        context.moveTo(pixel + 0.5, y1);
        context.lineTo(pixel + 0.5, y2);
        context.stroke();
      }
    }

    if (label) {
      context.fillStyle = "#f7f5ef";
      context.font = "600 12px IBM Plex Sans";
      context.fillText(label, 12, 18);
    }

    const duration = samples.length > 0 ? samples.length / sampleRate : 0;
    if (duration > 0) {
      context.fillStyle = "rgba(247, 245, 239, 0.65)";
      context.font = "11px JetBrains Mono";
      const steps = 5;

      for (let stepIndex = 0; stepIndex <= steps; stepIndex++) {
        const time = (duration / steps) * stepIndex;
        const x = (renderWidth / steps) * stepIndex;
        context.fillText(
          `${time.toFixed(1)}s`,
          Math.min(renderWidth - 34, x + 4),
          height - 8,
        );
      }
    }
  }, [amplitudeReference, color, height, label, renderWidth, sampleRate, samples]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: width ?? "100%", height, borderRadius: 8 }}
    />
  );
}
