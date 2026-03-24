import { useEffect, useState } from "react";
import type * as React from "react";
import {
  Mic,
  RotateCcw,
  Square,
} from "lucide-react";

import {
  AudioPlayer,
  AudioPlayerControlBar,
  AudioPlayerDurationDisplay,
  AudioPlayerElement,
  AudioPlayerMuteButton,
  AudioPlayerPlayButton,
  AudioPlayerSeekBackwardButton,
  AudioPlayerSeekForwardButton,
  AudioPlayerTimeDisplay,
  AudioPlayerTimeRange,
  AudioPlayerVolumeRange,
} from "@/components/ai-elements/audio-player";
import { Alert, AlertDescription, AlertTitle } from "../../components/ui/alert";
import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { Skeleton } from "../../components/ui/skeleton";
import { cn } from "../../lib/utils";
import type { AudioSamples } from "../audio";
import { SpectrogramCanvas, WaveformCanvas } from "../signal-views";
import { useSpeechWorkbench } from "./useSpeechWorkbench";

function InlineSpinner({
  className,
}: {
  readonly className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent",
        className,
      )}
    />
  );
}

function SpinnerLabel({ text }: { readonly text: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-zinc-400">
      <InlineSpinner />
      <span>{text}</span>
    </div>
  );
}

function StatRow({
  label,
  value,
  className,
}: {
  readonly label: string;
  readonly value: string;
  readonly className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between border-b border-zinc-800 py-2 last:border-b-0",
        className,
      )}
    >
      <dt className="text-sm text-zinc-400">{label}</dt>
      <dd className="text-right text-sm font-medium text-zinc-100">{value}</dd>
    </div>
  );
}

function CanvasShell({
  title,
  children,
}: {
  readonly title: string;
  readonly children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-zinc-200">{title}</h3>
      <div className="overflow-hidden rounded-md border border-zinc-800 bg-black">
        {children}
      </div>
    </div>
  );
}

function PanelShell({
  title,
  children,
}: {
  readonly title: string;
  readonly children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-zinc-200">{title}</h3>
      {children}
    </div>
  );
}

function EmptyChart({
  label,
  detail,
  height,
}: {
  readonly label: string;
  readonly detail?: string;
  readonly height: number;
}) {
  return (
    <div
      className="flex items-center justify-center rounded-md border border-dashed border-zinc-800 bg-zinc-950 px-6 text-center"
      style={{ height }}
    >
      <div className="space-y-1">
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        {detail ? <p className="text-sm text-zinc-500">{detail}</p> : null}
      </div>
    </div>
  );
}

function FixedStatPanel({
  children,
}: {
  readonly children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-md border border-zinc-800 bg-zinc-950 p-4"
      style={{ height: 180 }}
    >
      {children}
    </div>
  );
}

function PermissionPrompt() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6">
      <div className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-6 shadow-lg">
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full bg-zinc-900">
          <Mic className="size-5 text-zinc-100" />
        </div>
        <p className="text-center text-base font-medium text-zinc-100">
          Waiting for permissions
        </p>
        <p className="mt-2 text-center text-sm text-zinc-400">
          Grant site access to the microphone so it can record and calculate the
          fourier transform in real time.
        </p>
        <p className="mt-2 text-center text-sm text-zinc-500">
          All signal processing happens on your machine.
        </p>
        <div className="mt-4 flex items-center justify-center">
          <InlineSpinner className="size-5 text-zinc-400" />
        </div>
      </div>
    </div>
  );
}

const rms = (samples: Float32Array) => {
  if (samples.length === 0) {
    return 0;
  }

  let total = 0;
  for (let index = 0; index < samples.length; index++) {
    const value = samples[index] ?? 0;
    total += value * value;
  }

  return Math.sqrt(total / samples.length);
};

const peak = (samples: Float32Array) => {
  let max = 0;
  for (let index = 0; index < samples.length; index++) {
    max = Math.max(max, Math.abs(samples[index] ?? 0));
  }
  return max;
};

const toWavDataUrl = ({ samples, sampleRate }: AudioSamples) => {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index++) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index++) {
    const raw = samples[index] ?? 0;
    const sample = Number.isFinite(raw) ? Math.max(-1, Math.min(1, raw)) : 0;
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    const chunk = bytes.subarray(index, Math.min(index + chunkSize, bytes.length));
    binary += String.fromCharCode(...chunk);
  }

  return `data:audio/wav;base64,${btoa(binary)}`;
};

export function SpeechWorkbench() {
  const {
    state,
    presets,
    startRecording,
    stopRecording,
    reset,
    setSelectedEdit,
  } = useSpeechWorkbench();
  const [showOriginalReference, setShowOriginalReference] = useState(false);

  const liveAnalysis = state.live?.analysis;
  const ready = state.phase === "ready" && state.recorded && state.analysis;
  const hasRecording = state.recorded !== null;
  const starting = state.phase === "starting";
  const waitingForPermission =
    state.phase === "starting" && state.microphonePermission === "requesting";
  const liveEmptyLabel =
    state.phase === "recording" ? "Recording is in progress" : "No active recording";
  const liveEmptyDetail =
    state.phase === "recording"
      ? "Waiting for the first microphone frames to populate the live monitor."
      : "Start a recording to show the live signal and spectrogram.";
  const showPrimaryRecordButton =
    state.phase === "idle" ||
    state.phase === "ready" ||
    state.phase === "starting" ||
    state.phase === "analyzing";
  const hasActiveEdit = state.selectedEdit.type !== "identity";
  const selectedPreset = presets.find((preset) => preset.edit === state.selectedEdit);
  const selectedPresetLabel = selectedPreset?.label ?? "Unknown";
  const showEditedSignal =
    Boolean(ready) &&
    hasActiveEdit &&
    !showOriginalReference &&
    Boolean(state.edited);
  const signalSectionTitle =
    state.phase === "recording" || !hasRecording ? "Live monitoring" : "Analysis";
  const differenceRms = state.edited ? rms(state.edited.difference) : 0;
  const differencePeak = state.edited ? peak(state.edited.difference) : 0;
  const signalMode = state.phase === "analyzing"
    ? "busy"
    : state.phase === "recording" && state.live && liveAnalysis
      ? "live"
      : ready
        ? "analysis"
        : "empty";
  const editLoading =
    signalMode === "analysis" &&
    hasActiveEdit &&
    !showOriginalReference &&
    state.applyingEdit;
  const valuePanelTitle =
    signalMode === "analysis" && (showEditedSignal || editLoading)
      ? "Difference values"
      : "Session values";
  const displayedAudio =
    signalMode === "analysis"
      ? showEditedSignal && !editLoading && state.edited
        ? state.edited.audio
        : state.recorded
      : null;
  const [audioSrc, setAudioSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!hasActiveEdit && showOriginalReference) {
      setShowOriginalReference(false);
    }
  }, [hasActiveEdit, showOriginalReference]);

  useEffect(() => {
    setAudioSrc(displayedAudio ? toWavDataUrl(displayedAudio) : null);
  }, [displayedAudio]);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      {waitingForPermission && <PermissionPrompt />}

      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-6">
        <header className="border-b border-zinc-800 pb-4">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
            Speech spectrogram
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Record speech, inspect the live window, and compare spectral edits.
          </p>
        </header>

        <section className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Session</CardTitle>
                <CardDescription>
                  Start a recording to inspect the rolling window and final analysis.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {state.microphonePermission === "unsupported" && (
                  <Alert variant="destructive">
                    <AlertTitle>Microphone recording is not supported</AlertTitle>
                    <AlertDescription>
                      This browser does not expose the APIs needed for live capture or
                      recording.
                    </AlertDescription>
                  </Alert>
                )}

                <div className="flex flex-wrap items-center gap-3">
                  {showPrimaryRecordButton && (
                    <Button
                      onClick={startRecording}
                      disabled={state.phase === "starting" || state.phase === "analyzing"}
                      variant={state.phase === "analyzing" ? "secondary" : "default"}
                      className={
                        state.phase === "analyzing"
                          ? "min-w-[168px] bg-sky-900 text-sky-50 hover:bg-sky-900"
                          : "min-w-[168px]"
                      }
                    >
                      {state.phase === "starting" ? (
                        <>
                          <InlineSpinner />
                          {waitingForPermission ? "Waiting for permissions" : "Start recording"}
                        </>
                      ) : state.phase === "analyzing" ? (
                        <>
                          <InlineSpinner />
                          Processing
                        </>
                      ) : (
                        <>
                          <Mic className="size-4" />
                          Start recording
                        </>
                      )}
                    </Button>
                  )}

                  {state.phase === "recording" && (
                    <Button
                      variant="destructive"
                      onClick={stopRecording}
                      className="min-w-[168px]"
                    >
                      <Square className="size-4" />
                      Stop recording
                    </Button>
                  )}

                  {hasRecording && (
                    <>
                      <Button variant="outline" onClick={reset}>
                        <RotateCcw className="size-4" />
                        Reset
                      </Button>
                    </>
                  )}
                </div>

                {state.error && (
                  <Alert variant="destructive">
                    <AlertTitle>Recording error</AlertTitle>
                    <AlertDescription>{state.error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle>{signalSectionTitle}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="grid gap-4 md:grid-cols-2">
                  <CanvasShell title="Waveform">
                    {signalMode === "busy" ? (
                      <Skeleton className="h-[180px] w-full" />
                    ) : starting ? (
                      <EmptyChart
                        label={liveEmptyLabel}
                        detail={liveEmptyDetail}
                        height={180}
                      />
                    ) : signalMode === "live" ? (
                      <WaveformCanvas
                        samples={state.live!.frame.samples}
                        sampleRate={state.live!.frame.sampleRate}
                        label="Live waveform"
                        color="#00d4aa"
                        amplitudeReference={state.live!.frame.peakAmplitude}
                        className="w-full"
                      />
                    ) : signalMode === "analysis" ? (
                      editLoading ? (
                        <Skeleton className="h-[180px] w-full" />
                      ) : (
                        <WaveformCanvas
                          samples={
                            showEditedSignal
                              ? state.edited!.audio.samples
                              : state.recorded!.samples
                          }
                          sampleRate={
                            showEditedSignal
                              ? state.edited!.audio.sampleRate
                              : state.recorded!.sampleRate
                          }
                          label={showEditedSignal ? "Edited waveform" : "Original waveform"}
                          color={showEditedSignal ? "#ff9d5c" : "#00d4aa"}
                          className="w-full"
                        />
                      )
                    ) : (
                      <EmptyChart
                        label={liveEmptyLabel}
                        detail={liveEmptyDetail}
                        height={180}
                      />
                    )}
                  </CanvasShell>

                  <PanelShell title={valuePanelTitle}>
                    <FixedStatPanel>
                      {signalMode === "busy" ? (
                        <div className="space-y-3">
                          <Skeleton className="h-4 w-24" />
                          <Skeleton className="h-4 w-36" />
                          <Skeleton className="h-4 w-28" />
                          <Skeleton className="h-4 w-32" />
                          <Skeleton className="h-4 w-40" />
                        </div>
                      ) : starting ? (
                        <div className="flex h-full items-center justify-center text-center">
                          <div className="space-y-1">
                            <p className="text-sm font-medium text-zinc-200">
                              {liveEmptyLabel}
                            </p>
                            <p className="text-sm text-zinc-500">
                              Session values appear after recording starts or finishes.
                            </p>
                          </div>
                        </div>
                      ) : signalMode === "live" ? (
                        <dl className="grid h-full grid-rows-5">
                          <StatRow
                            label="Window"
                            value={`${(state.live!.frame.elapsedMs / 1000).toFixed(1)} s`}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Input level"
                            value={`${(state.live!.frame.level * 100).toFixed(1)} %`}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Session peak"
                            value={`${(state.live!.frame.peakAmplitude * 100).toFixed(1)} %`}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Pitch"
                            value={
                              liveAnalysis!.medianF0
                                ? `${liveAnalysis!.medianF0.toFixed(1)} Hz`
                                : "Unvoiced"
                            }
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Formants"
                            value={
                              liveAnalysis!.formantMedians.length > 0
                                ? liveAnalysis!.formantMedians
                                    .map((value) => Math.round(value))
                                    .join(" / ")
                                : "Tracking"
                            }
                            className="h-full py-0"
                          />
                        </dl>
                      ) : signalMode === "analysis" && editLoading ? (
                        <div className="space-y-3">
                          <Skeleton className="h-4 w-24" />
                          <Skeleton className="h-4 w-36" />
                          <Skeleton className="h-4 w-28" />
                          <Skeleton className="h-4 w-32" />
                          <Skeleton className="h-4 w-40" />
                        </div>
                      ) : signalMode === "analysis" && showEditedSignal && state.edited ? (
                        <dl className="grid h-full grid-rows-5">
                          <StatRow
                            label="Edit"
                            value={selectedPresetLabel}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="RMS delta"
                            value={`${(differenceRms * 100).toFixed(1)} %`}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Peak delta"
                            value={`${(differencePeak * 100).toFixed(1)} %`}
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Edited pitch"
                            value={
                              state.edited.analysis.medianF0
                                ? `${state.edited.analysis.medianF0.toFixed(1)} Hz`
                                : "Unvoiced"
                            }
                            className="h-full py-0"
                          />
                          <StatRow
                            label="Edited formants"
                            value={
                              state.edited.analysis.formantMedians.length > 0
                                ? state.edited.analysis.formantMedians
                                    .map((value) => Math.round(value))
                                    .join(" / ")
                                : "Unavailable"
                            }
                            className="h-full py-0"
                          />
                        </dl>
                      ) : signalMode === "analysis" ? (
                        <dl>
                          <StatRow
                            label="Duration"
                            value={`${(
                              state.recorded!.samples.length / state.recorded!.sampleRate
                            ).toFixed(2)} s`}
                          />
                          <StatRow
                            label="Frames"
                            value={String(state.analysis!.stft.frames.length)}
                          />
                          <StatRow
                            label="Pitch"
                            value={
                              state.analysis!.medianF0
                                ? `${state.analysis!.medianF0.toFixed(1)} Hz`
                                : "Unvoiced"
                            }
                          />
                          <StatRow
                            label="Formants"
                            value={
                              state.analysis!.formantMedians.length > 0
                                ? state.analysis!.formantMedians
                                    .map((value) => Math.round(value))
                                    .join(" / ")
                                : "Unavailable"
                            }
                          />
                        </dl>
                      ) : (
                        <div className="flex h-full items-center justify-center text-center">
                          <div className="space-y-1">
                            <p className="text-sm font-medium text-zinc-200">
                              {liveEmptyLabel}
                            </p>
                            <p className="text-sm text-zinc-500">
                              Session values appear after recording starts or finishes.
                            </p>
                          </div>
                        </div>
                      )}
                    </FixedStatPanel>
                  </PanelShell>
                </div>

                <CanvasShell title="Spectrogram">
                  {signalMode === "busy" ? (
                    <Skeleton className="h-[320px] w-full" />
                  ) : starting ? (
                    <EmptyChart
                      label={liveEmptyLabel}
                      detail="The spectrogram appears during recording and stays here after processing."
                      height={320}
                    />
                  ) : signalMode === "live" ? (
                    <SpectrogramCanvas
                      stft={liveAnalysis!.stft}
                      pitchTrack={liveAnalysis!.pitchTrack}
                      formants={liveAnalysis!.formants}
                      maxFreqDisplay={Math.min(state.live!.frame.sampleRate / 2, 8000)}
                      className="w-full"
                    />
                  ) : signalMode === "analysis" ? (
                    editLoading ? (
                      <Skeleton className="h-[320px] w-full" />
                    ) : (
                      <SpectrogramCanvas
                        stft={
                          showEditedSignal
                            ? state.edited!.analysis.stft
                            : state.analysis!.stft
                        }
                        pitchTrack={
                          showEditedSignal
                            ? state.edited!.analysis.pitchTrack
                            : state.analysis!.pitchTrack
                        }
                        formants={
                          showEditedSignal
                            ? state.edited!.analysis.formants
                            : state.analysis!.formants
                        }
                        maxFreqDisplay={Math.min(state.recorded!.sampleRate / 2, 8000)}
                        className="w-full"
                      />
                    )
                  ) : (
                    <EmptyChart
                      label={liveEmptyLabel}
                      detail="The spectrogram appears during recording and stays here after processing."
                      height={320}
                    />
                  )}
                </CanvasShell>

                <PanelShell title="Audio">
                  <div className="rounded-md border border-zinc-800 bg-zinc-950 p-4">
                    {signalMode === "busy" ? (
                      <div className="flex h-[40px] items-center">
                        <Skeleton className="h-9 w-full" />
                      </div>
                    ) : editLoading ? (
                      <div className="flex h-[40px] items-center">
                        <Skeleton className="h-9 w-full" />
                      </div>
                    ) : audioSrc ? (
                      <div className="flex h-[40px] items-center">
                        <AudioPlayer className="w-full">
                          <AudioPlayerElement
                            key={audioSrc}
                            src={audioSrc}
                            preload="metadata"
                          />
                          <AudioPlayerControlBar className="flex w-full flex-wrap items-center gap-2">
                            <AudioPlayerPlayButton />
                            <AudioPlayerSeekBackwardButton />
                            <AudioPlayerSeekForwardButton />
                            <AudioPlayerTimeDisplay showDuration={false} />
                            <AudioPlayerTimeRange className="min-w-[160px] flex-1" />
                            <AudioPlayerDurationDisplay />
                            <AudioPlayerMuteButton />
                            <AudioPlayerVolumeRange className="w-20" />
                          </AudioPlayerControlBar>
                        </AudioPlayer>
                      </div>
                    ) : (
                      <div className="flex min-h-20 items-center justify-center text-center">
                        <div className="space-y-1">
                          <p className="text-sm font-medium text-zinc-200">No audio loaded</p>
                          <p className="text-sm text-zinc-500">
                            The player appears once a processed recording is available.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </PanelShell>

                <PanelShell title="Edit controls">
                  <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950 p-4">
                    <div className="flex flex-wrap gap-2">
                      {presets.map((preset) => {
                        const active = selectedPreset?.id === preset.id;
                        return (
                          <Button
                            key={preset.id}
                            variant={active ? "secondary" : "outline"}
                            size="sm"
                            disabled={signalMode !== "analysis"}
                            onClick={() => setSelectedEdit(preset.edit)}
                          >
                            {preset.label}
                          </Button>
                        );
                      })}
                    </div>

                    <div className="flex flex-wrap items-center gap-3">
                      <Button
                        variant={showOriginalReference ? "secondary" : "outline"}
                        size="sm"
                        disabled={signalMode !== "analysis" || !hasActiveEdit}
                        onClick={() => setShowOriginalReference((current) => !current)}
                      >
                        {showOriginalReference ? "Showing original" : "Show original"}
                      </Button>
                      {state.applyingEdit && <SpinnerLabel text="Applying spectral edit." />}
                    </div>
                  </div>
                </PanelShell>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}
