import type { PropsWithChildren } from "react";

import { SpectrogramCanvas, WaveformCanvas } from "../signal-views";
import { useSpeechWorkbench } from "./useSpeechWorkbench";

function Surface({
  title,
  eyebrow,
  children,
}: PropsWithChildren<{ readonly title: string; readonly eyebrow?: string }>) {
  return (
    <section
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 24,
        padding: 20,
        boxShadow: "var(--shadow)",
        backdropFilter: "blur(18px)",
      }}
    >
      <div style={{ marginBottom: 14 }}>
        {eyebrow ? (
          <div
            style={{
              color: "var(--accent)",
              fontSize: 12,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              marginBottom: 4,
            }}
          >
            {eyebrow}
          </div>
        ) : null}
        <h2 style={{ margin: 0, fontSize: 24 }}>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div
      style={{
        padding: 14,
        borderRadius: 16,
        background: "rgba(255, 255, 255, 0.04)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600 }}>{value}</div>
    </div>
  );
}

function ActionButton({
  label,
  onClick,
  disabled,
  tone = "primary",
}: {
  readonly label: string;
  readonly onClick: () => void;
  readonly disabled?: boolean;
  readonly tone?: "primary" | "danger" | "secondary";
}) {
  const background =
    tone === "danger"
      ? "linear-gradient(135deg, #c44141, #ff6b6b)"
      : tone === "secondary"
        ? "rgba(255, 255, 255, 0.08)"
        : "linear-gradient(135deg, #3fb5ad, #54d4c4)";
  const color = tone === "secondary" ? "var(--text)" : "#061013";

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        padding: "12px 18px",
        borderRadius: 999,
        background,
        color,
        fontWeight: 700,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.55 : 1,
      }}
    >
      {label}
    </button>
  );
}

export function SpeechWorkbench() {
  const {
    state,
    presets,
    startRecording,
    stopRecording,
    reset,
    setSelectedEdit,
    playOriginal,
    playEdited,
  } = useSpeechWorkbench();

  const liveAnalysis = state.live?.analysis;
  const ready = state.phase === "ready" && state.recorded && state.analysis;

  return (
    <main
      style={{
        width: "min(1180px, calc(100vw - 32px))",
        margin: "0 auto",
        padding: "40px 0 56px",
      }}
    >
      <section style={{ marginBottom: 28 }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 12px",
            borderRadius: 999,
            border: "1px solid var(--border)",
            background: "rgba(9, 18, 23, 0.55)",
            marginBottom: 16,
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: state.phase === "recording" ? "var(--danger)" : "var(--accent)",
              boxShadow:
                state.phase === "recording"
                  ? "0 0 18px rgba(255, 107, 107, 0.8)"
                  : "0 0 18px rgba(84, 212, 196, 0.6)",
            }}
          />
          <span style={{ color: "var(--muted)", fontSize: 13 }}>
            {state.phase === "recording"
              ? "Live microphone capture"
              : state.phase === "analyzing"
                ? "Rendering final analysis"
                : "Effect beta orchestration"}
          </span>
        </div>

        <h1
          style={{
            fontSize: "clamp(2.8rem, 5vw, 5.2rem)",
            lineHeight: 0.95,
            letterSpacing: "-0.05em",
            margin: "0 0 12px",
            maxWidth: 900,
          }}
        >
          Speech spectrogram workbench
        </h1>

        <p style={{ margin: 0, color: "var(--muted)", maxWidth: 760, fontSize: 18 }}>
          The UI is a thin shell now. Browser audio, DSP analysis, and playback live behind
          explicit Effect services, while the workbench module owns the orchestration boundary.
        </p>
      </section>

      <Surface title="Session Controls" eyebrow="Capture">
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          {state.phase === "idle" || state.phase === "ready" ? (
            <ActionButton label="Start Recording" onClick={startRecording} />
          ) : null}

          {state.phase === "recording" ? (
            <ActionButton label="Stop Recording" tone="danger" onClick={stopRecording} />
          ) : null}

          {state.phase !== "idle" ? (
            <ActionButton label="Reset" tone="secondary" onClick={reset} />
          ) : null}

          {ready ? (
            <>
              <ActionButton
                label={state.playing === "original" ? "Playing Original" : "Play Original"}
                onClick={playOriginal}
                disabled={state.playing !== null}
                tone="secondary"
              />
              <ActionButton
                label={state.playing === "edited" ? "Playing Edited" : "Play Edited"}
                onClick={playEdited}
                disabled={state.playing !== null || !state.edited}
                tone="secondary"
              />
            </>
          ) : null}
        </div>

        <div style={{ marginTop: 16, color: "var(--muted)", fontSize: 14 }}>
          {state.phase === "recording"
            ? "Live previews update continuously while MediaRecorder captures the final take."
            : state.phase === "analyzing"
              ? "Finalizing the recording, decoding it, and recomputing the full STFT."
              : "Record a phrase to inspect waveform, pitch, formants, and spectral edits."}
        </div>

        {state.error ? (
          <div
            style={{
              marginTop: 16,
              borderRadius: 16,
              padding: "12px 14px",
              background: "rgba(196, 65, 65, 0.12)",
              border: "1px solid rgba(255, 107, 107, 0.3)",
              color: "#ffd8d8",
            }}
          >
            {state.error}
          </div>
        ) : null}
      </Surface>

      {state.live && liveAnalysis ? (
        <div style={{ display: "grid", gap: 20, gridTemplateColumns: "1fr", marginTop: 24 }}>
          <Surface title="Live Monitor" eyebrow="Realtime">
            <div
              style={{
                display: "grid",
                gap: 14,
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                marginBottom: 18,
              }}
            >
              <Metric label="Window" value={`${(state.live.frame.elapsedMs / 1000).toFixed(1)}s`} />
              <Metric label="Input Level" value={`${(state.live.frame.level * 100).toFixed(1)}%`} />
              <Metric
                label="Pitch"
                value={liveAnalysis.medianF0 ? `${liveAnalysis.medianF0.toFixed(1)} Hz` : "Unvoiced"}
              />
              <Metric
                label="Formants"
                value={
                  liveAnalysis.formantMedians.length > 0
                    ? liveAnalysis.formantMedians.map((value) => Math.round(value)).join(" / ")
                    : "Tracking"
                }
              />
            </div>

            <div style={{ display: "grid", gap: 18 }}>
              <div>
                <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                  Rolling time-domain window
                </div>
                <WaveformCanvas
                  samples={state.live.frame.samples}
                  sampleRate={state.live.frame.sampleRate}
                  label="Live waveform"
                  color="#54d4c4"
                  amplitudeReference={state.live.frame.peakAmplitude}
                />
              </div>

              <div>
                <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                  Streaming STFT, pitch track, and formant hints
                </div>
                <SpectrogramCanvas
                  stft={liveAnalysis.stft}
                  pitchTrack={liveAnalysis.pitchTrack}
                  formants={liveAnalysis.formants}
                  maxFreqDisplay={Math.min(state.live.frame.sampleRate / 2, 8000)}
                />
              </div>
            </div>
          </Surface>
        </div>
      ) : null}

      {ready ? (
        <div style={{ display: "grid", gap: 24, marginTop: 24 }}>
          <Surface title="Recording Analysis" eyebrow="Overview">
            <div
              style={{
                display: "grid",
                gap: 14,
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                marginBottom: 18,
              }}
            >
              <Metric
                label="Pitch"
                value={state.analysis.medianF0 ? `${state.analysis.medianF0.toFixed(1)} Hz` : "Unvoiced"}
              />
              <Metric label="Frames" value={String(state.analysis.stft.frames.length)} />
              <Metric
                label="Duration"
                value={`${(state.recorded.samples.length / state.recorded.sampleRate).toFixed(2)}s`}
              />
              <Metric
                label="Formants"
                value={
                  state.analysis.formantMedians.length > 0
                    ? state.analysis.formantMedians.map((value) => Math.round(value)).join(" / ")
                    : "Unavailable"
                }
              />
            </div>

            <div style={{ display: "grid", gap: 18 }}>
              <div>
                <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                  Original waveform
                </div>
                <WaveformCanvas
                  samples={state.recorded.samples}
                  sampleRate={state.recorded.sampleRate}
                  label="Original signal"
                  color="#54d4c4"
                />
              </div>

              <div>
                <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                  Original STFT with pitch and formants
                </div>
                <SpectrogramCanvas
                  stft={state.analysis.stft}
                  pitchTrack={state.analysis.pitchTrack}
                  formants={state.analysis.formants}
                  maxFreqDisplay={Math.min(state.recorded.sampleRate / 2, 8000)}
                />
              </div>
            </div>
          </Surface>

          <Surface title="Spectral Edit Module" eyebrow="Experiment">
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 18 }}>
              {presets.map((preset) => {
                const active = JSON.stringify(preset.edit) === JSON.stringify(state.selectedEdit);
                return (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => setSelectedEdit(preset.edit)}
                    style={{
                      padding: "10px 14px",
                      borderRadius: 999,
                      border: active ? "1px solid rgba(84, 212, 196, 0.8)" : "1px solid rgba(255, 255, 255, 0.08)",
                      background: active ? "rgba(84, 212, 196, 0.14)" : "rgba(255, 255, 255, 0.04)",
                      color: "var(--text)",
                      cursor: "pointer",
                    }}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>

            {state.edited ? (
              <div style={{ display: "grid", gap: 18 }}>
                <div style={{ display: "grid", gap: 18, gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
                  <div>
                    <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                      Edited waveform
                    </div>
                    <WaveformCanvas
                      samples={state.edited.audio.samples}
                      sampleRate={state.edited.audio.sampleRate}
                      label="Edited signal"
                      color="#ff9d5c"
                      width={540}
                    />
                  </div>
                  <div>
                    <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                      Difference signal
                    </div>
                    <WaveformCanvas
                      samples={state.edited.difference}
                      sampleRate={state.recorded.sampleRate}
                      label="Original - edited"
                      color="#7aa2ff"
                      width={540}
                    />
                  </div>
                </div>

                <div>
                  <div style={{ marginBottom: 10, color: "var(--muted)", fontSize: 13 }}>
                    Edited STFT
                  </div>
                  <SpectrogramCanvas
                    stft={state.edited.analysis.stft}
                    pitchTrack={state.edited.analysis.pitchTrack}
                    formants={state.edited.analysis.formants}
                    maxFreqDisplay={Math.min(state.recorded.sampleRate / 2, 8000)}
                  />
                </div>
              </div>
            ) : null}
          </Surface>
        </div>
      ) : null}
    </main>
  );
}
