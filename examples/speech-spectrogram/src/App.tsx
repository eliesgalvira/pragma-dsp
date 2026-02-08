import { useState, useCallback, useMemo } from "react";
import {
  getUserMicrophoneStream,
  recordStreamAsBlob,
  blobToFloat32Array,
  playSignal,
} from "./audio/record";
import { computeStft } from "./dsp/stft";
import { detectPitch, detectFormants } from "./dsp/pitch";
import { WaveformCanvas } from "./components/WaveformCanvas";
import { SpectrogramCanvas } from "./components/SpectrogramCanvas";
import { SpectrogramEditor } from "./components/SpectrogramEditor";

const SAMPLE_RATE = 16_000;
const FFT_SIZE = 1024;
const HOP_SIZE = 256;

type RecordingState =
  | { phase: "idle" }
  | { phase: "recording"; stop: () => Promise<Blob> }
  | { phase: "decoding" }
  | {
      phase: "done";
      samples: Float32Array;
      sampleRate: number;
    };

export default function App() {
  const [state, setState] = useState<RecordingState>({ phase: "idle" });
  const [error, setError] = useState<string | null>(null);
  const [playingOriginal, setPlayingOriginal] = useState(false);

  // ── Recording controls ───────────────────────────────────────────
  const handleStart = useCallback(async () => {
    try {
      setError(null);
      const stream = await getUserMicrophoneStream();
      const stop = recordStreamAsBlob(stream);
      setState({ phase: "recording", stop });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const handleStop = useCallback(async () => {
    if (state.phase !== "recording") return;
    setState({ phase: "decoding" });
    try {
      const blob = await state.stop();
      const { samples, sampleRate } = await blobToFloat32Array(blob, SAMPLE_RATE);
      setState({ phase: "done", samples, sampleRate });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState({ phase: "idle" });
    }
  }, [state]);

  const handleReset = useCallback(() => {
    setState({ phase: "idle" });
    setError(null);
  }, []);

  // ── Computed DSP results (only when recording is done) ───────────
  const analysis = useMemo(() => {
    if (state.phase !== "done") return null;
    const { samples, sampleRate } = state;

    // STFT for spectrogram
    const stft = computeStft(samples, {
      fftSize: FFT_SIZE,
      hopSize: HOP_SIZE,
      sampleRate,
      window: "hann",
    });

    // Pitch track (per frame)
    const pitchTrack: (number | null)[] = [];
    for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP_SIZE) {
      const frame = samples.subarray(start, start + FFT_SIZE);
      const { f0 } = detectPitch(frame, sampleRate);
      pitchTrack.push(f0);
    }

    // Formants (per frame, using STFT magnitudes + frequencies)
    const formants = stft.frames.map((f) =>
      detectFormants(f.magnitudes, stft.frequencies, {
        smoothingWidth: 15,
        maxFormants: 4,
      })
    );

    // Overall pitch (median of non-null)
    const validPitches = pitchTrack.filter((p): p is number => p != null && p > 0);
    validPitches.sort((a, b) => a - b);
    const medianF0 =
      validPitches.length > 0
        ? validPitches[Math.floor(validPitches.length / 2)]!
        : null;

    // Overall formants (median of each formant across frames)
    const formantMedians: number[] = [];
    for (let fi = 0; fi < 4; fi++) {
      const vals = formants
        .map((f) => f.formants[fi])
        .filter((v): v is number => v != null && v > 0);
      vals.sort((a, b) => a - b);
      if (vals.length > 0) {
        formantMedians.push(vals[Math.floor(vals.length / 2)]!);
      }
    }

    return { stft, pitchTrack, formants, medianF0, formantMedians };
  }, [state]);

  const handlePlayOriginal = async () => {
    if (state.phase !== "done") return;
    setPlayingOriginal(true);
    try {
      await playSignal(state.samples, state.sampleRate);
    } finally {
      setPlayingOriginal(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif", color: "#e0e0e0" }}>
      <h1 style={{ marginBottom: 4 }}>🎙️ Speech Spectrogram</h1>
      <p style={{ color: "#999", marginTop: 0 }}>
        Powered by{" "}
        <code style={{ color: "#4fc3f7" }}>pragma-dsp</code> — record, analyze, edit spectra, hear the difference.
      </p>

      {error && (
        <div style={{ background: "#4a1c1c", padding: "8px 14px", borderRadius: 4, marginBottom: 12 }}>
          ⚠️ {error}
        </div>
      )}

      {/* ── Recording controls ──────────────────────────────────── */}
      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        {state.phase === "idle" && (
          <button onClick={handleStart} style={btnStyle}>
            🎤 Start Recording
          </button>
        )}
        {state.phase === "recording" && (
          <button onClick={handleStop} style={{ ...btnStyle, background: "#c62828" }}>
            ⏹ Stop Recording
          </button>
        )}
        {state.phase === "decoding" && (
          <button disabled style={btnStyle}>
            ⏳ Decoding…
          </button>
        )}
        {state.phase === "done" && (
          <>
            <button onClick={handlePlayOriginal} disabled={playingOriginal} style={btnStyle}>
              {playingOriginal ? "▶ Playing…" : "▶ Play original"}
            </button>
            <button onClick={handleReset} style={{ ...btnStyle, background: "#37474f" }}>
              🔄 New Recording
            </button>
          </>
        )}
      </div>

      {state.phase === "recording" && (
        <div style={{ padding: 20, textAlign: "center", color: "#ef5350", fontSize: 18, fontWeight: "bold" }}>
          ● Recording… speak now
        </div>
      )}

      {/* ── Analysis results ────────────────────────────────────── */}
      {state.phase === "done" && analysis && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Waveform */}
          <section>
            <h3 style={{ margin: "0 0 6px" }}>Waveform</h3>
            <WaveformCanvas
              samples={state.samples}
              sampleRate={state.sampleRate}
              label="Original signal"
            />
          </section>

          {/* Pitch & formant summary */}
          <section style={{ background: "#1a1a2e", padding: "12px 16px", borderRadius: 6 }}>
            <h3 style={{ margin: "0 0 8px" }}>Pitch & Formants</h3>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
              <div>
                <strong>F0 (pitch):</strong>{" "}
                {analysis.medianF0
                  ? `${analysis.medianF0.toFixed(1)} Hz`
                  : "—"}
              </div>
              {analysis.formantMedians.map((f, i) => (
                <div key={i}>
                  <strong>F{i + 1}:</strong> {f.toFixed(0)} Hz
                </div>
              ))}
            </div>
            <p style={{ color: "#777", fontSize: 12, margin: "8px 0 0" }}>
              F0 via autocorrelation (Wiener–Khinchin). Formants via spectral envelope peak-picking.
            </p>
          </section>

          {/* Spectrogram */}
          <section>
            <h3 style={{ margin: "0 0 6px" }}>
              Spectrogram{" "}
              <span style={{ fontWeight: "normal", fontSize: 12, color: "#999" }}>
                (orange = F0 track, dots = formants)
              </span>
            </h3>
            <SpectrogramCanvas
              stft={analysis.stft}
              pitchTrack={analysis.pitchTrack}
              formants={analysis.formants}
              width={800}
              height={300}
              maxFreqDisplay={Math.min(state.sampleRate / 2, 8000)}
            />
          </section>

          {/* Spectrogram editor */}
          <section style={{ borderTop: "1px solid #333", paddingTop: 16 }}>
            <SpectrogramEditor
              samples={state.samples}
              sampleRate={state.sampleRate}
              fftSize={FFT_SIZE}
            />
          </section>
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "10px 20px",
  borderRadius: 6,
  border: "1px solid #555",
  background: "#263238",
  color: "#e0e0e0",
  cursor: "pointer",
  fontSize: 14,
  fontWeight: "bold",
};
