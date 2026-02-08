import { useState, useMemo } from "react";
import { applySpectralEdit, EDIT_PRESETS, type EditKind } from "../dsp/edits";
import { computeStft } from "../dsp/stft";
import { WaveformCanvas } from "./WaveformCanvas";
import { SpectrogramCanvas } from "./SpectrogramCanvas";
import { playSignal } from "../audio/record";

type Props = {
  /** Original time-domain samples */
  samples: Float32Array;
  sampleRate: number;
  fftSize: number;
};

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Spectrogram Editor panel.
 *
 * Shows the original waveform, lets the user pick a spectral edit,
 * applies it via FluentFFT, and shows the reconstructed signal
 * side-by-side with the original.
 */
export function SpectrogramEditor({ samples, sampleRate, fftSize }: Props) {
  const [selectedEdit, setSelectedEdit] = useState<EditKind>(
    EDIT_PRESETS[0]!.edit
  );
  const [playing, setPlaying] = useState<"original" | "edited" | null>(null);

  // Apply the selected edit to the entire signal (frame-by-frame, then overlap-add)
  const { editedSignal, editedStft } = useMemo(() => {
    // For simplicity, process the entire signal as one FFT frame
    // (zero-padded to next power of 2)
    const paddedSize = nextPow2(samples.length);
    const input = new Float64Array(paddedSize);
    for (let i = 0; i < samples.length; i++) input[i] = samples[i]!;

    const { edited, editedComplex } = applySpectralEdit(
      input,
      paddedSize,
      selectedEdit
    );

    // Trim to original length
    const trimmed = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) trimmed[i] = edited[i]!;

    // Compute STFT of edited signal for spectrogram display
    const stft = computeStft(trimmed, {
      fftSize,
      sampleRate,
      window: "hann",
    });

    return { editedSignal: trimmed, editedStft: stft };
  }, [samples, sampleRate, fftSize, selectedEdit]);

  // Compute difference signal
  const diffSignal = useMemo(() => {
    const diff = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      diff[i] = samples[i]! - editedSignal[i]!;
    }
    return diff;
  }, [samples, editedSignal]);

  const handlePlay = async (which: "original" | "edited") => {
    setPlaying(which);
    try {
      const sig = which === "original" ? samples : editedSignal;
      await playSignal(sig, sampleRate);
    } finally {
      setPlaying(null);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <h3 style={{ margin: 0, color: "#e0e0e0" }}>🎛️ Spectrogram Editor</h3>

      {/* Edit selector */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {EDIT_PRESETS.map((preset) => (
          <button
            key={preset.label}
            onClick={() => setSelectedEdit(preset.edit)}
            style={{
              padding: "6px 14px",
              borderRadius: 4,
              border:
                JSON.stringify(preset.edit) === JSON.stringify(selectedEdit)
                  ? "2px solid #4fc3f7"
                  : "1px solid #555",
              background:
                JSON.stringify(preset.edit) === JSON.stringify(selectedEdit)
                  ? "#263238"
                  : "#1a1a2e",
              color: "#e0e0e0",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Side-by-side waveforms */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <div>
          <WaveformCanvas
            samples={editedSignal}
            sampleRate={sampleRate}
            label="Edited signal"
            color="#ff8a65"
            width={390}
            height={120}
          />
          <button
            onClick={() => handlePlay("edited")}
            disabled={playing !== null}
            style={playBtnStyle}
          >
            {playing === "edited" ? "▶ Playing…" : "▶ Play edited"}
          </button>
        </div>
        <div>
          <WaveformCanvas
            samples={diffSignal}
            sampleRate={sampleRate}
            label="Difference (original − edited)"
            color="#ce93d8"
            width={390}
            height={120}
          />
        </div>
      </div>

      {/* Edited spectrogram */}
      <SpectrogramCanvas
        stft={editedStft}
        width={800}
        height={220}
        maxFreqDisplay={Math.min(sampleRate / 2, 8000)}
      />
    </div>
  );
}

const playBtnStyle: React.CSSProperties = {
  marginTop: 6,
  padding: "4px 12px",
  borderRadius: 4,
  border: "1px solid #555",
  background: "#263238",
  color: "#e0e0e0",
  cursor: "pointer",
  fontSize: 12,
};
