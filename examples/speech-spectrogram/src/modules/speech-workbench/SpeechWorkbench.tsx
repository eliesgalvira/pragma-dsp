import type * as React from "react";
import {
  AlertTriangle,
  LoaderCircle,
  Mic,
  RotateCcw,
  Square,
  Volume2,
  WandSparkles,
} from "lucide-react";

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "../../components/ui/alert";
import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { Skeleton } from "../../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import { cn } from "../../lib/utils";
import { SpectrogramCanvas, WaveformCanvas } from "../signal-views";
import { useSpeechWorkbench } from "./useSpeechWorkbench";

function SpinnerLabel({ text }: { readonly text: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-zinc-400">
      <LoaderCircle className="size-4 animate-spin" />
      <span>{text}</span>
    </div>
  );
}

function StatRow({
  label,
  value,
}: {
  readonly label: string;
  readonly value: string;
}) {
  return (
    <div className="flex items-center justify-between border-b border-zinc-800 py-2 last:border-b-0">
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

function AnalysisSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
      <Skeleton className="h-[220px] w-full" />
      <Skeleton className="h-[300px] w-full" />
    </div>
  );
}

function EmptyChart({
  label,
  detail,
}: {
  readonly label: string;
  readonly detail?: string;
}) {
  return (
    <div className="flex h-[220px] items-center justify-center rounded-md border border-dashed border-zinc-800 bg-zinc-950 px-6 text-center">
      <div className="space-y-1">
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        {detail ? <p className="text-sm text-zinc-500">{detail}</p> : null}
      </div>
    </div>
  );
}

function PermissionPrompt() {
  return (
    <div className="flex h-[220px] items-center justify-center rounded-md border border-zinc-800 bg-zinc-950 px-6">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full bg-zinc-800">
          <Mic className="size-5 text-zinc-100" />
        </div>
        <p className="text-base font-medium text-zinc-100">Waiting for permissions</p>
        <p className="mt-2 text-sm text-zinc-400">
          Grant site access to the microphone so it can record and calculate the
          fourier transform in real time.
        </p>
      </div>
    </div>
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
  const hasRecording = state.recorded !== null;
  const busy = state.phase === "starting" || state.phase === "analyzing";
  const showPermissionWarning =
    state.microphonePermission === "prompt" ||
    state.microphonePermission === "requesting" ||
    state.microphonePermission === "denied";
  const permissionWarningTitle =
    state.microphonePermission === "denied"
      ? "Microphone access is blocked"
      : "Give the site permission to access the microphone";
  const permissionWarningBody =
    state.microphonePermission === "denied"
      ? "Enable microphone access in the browser and try recording again. All signal processing happens on your machine."
      : "All signal processing happens on your machine.";

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-6">
        <header className="border-b border-zinc-800 pb-4">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
            Speech spectrogram
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Record speech, inspect the live window, and compare spectral edits.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Session</CardTitle>
                <CardDescription>
                  Start a recording to inspect the rolling window and final analysis.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {showPermissionWarning && (
                  <Alert variant="warning">
                    <AlertTitle>{permissionWarningTitle}</AlertTitle>
                    <AlertDescription>{permissionWarningBody}</AlertDescription>
                  </Alert>
                )}

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
                  {(state.phase === "idle" || state.phase === "ready") && (
                    <Button onClick={startRecording} disabled={busy}>
                      {busy ? (
                        <LoaderCircle className="size-4 animate-spin" />
                      ) : (
                        <Mic className="size-4" />
                      )}
                      Start recording
                    </Button>
                  )}

                  {state.phase === "recording" && (
                    <Button variant="destructive" onClick={stopRecording}>
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

                      <Button
                        variant="outline"
                        onClick={playOriginal}
                        disabled={!ready || state.playing !== null}
                      >
                        {state.playing === "original" ? (
                          <LoaderCircle className="size-4 animate-spin" />
                        ) : (
                          <Volume2 className="size-4" />
                        )}
                        Play original
                      </Button>

                      <Button
                        variant="outline"
                        onClick={playEdited}
                        disabled={
                          !ready ||
                          !state.edited ||
                          state.playing !== null ||
                          state.applyingEdit
                        }
                      >
                        {state.playing === "edited" ? (
                          <LoaderCircle className="size-4 animate-spin" />
                        ) : (
                          <WandSparkles className="size-4" />
                        )}
                        Play edited
                      </Button>
                    </>
                  )}
                </div>

                <div className="text-sm text-zinc-400">
                  {state.phase === "starting" && "Requesting microphone access."}
                  {state.phase === "recording" && "Recording in progress."}
                  {state.phase === "analyzing" &&
                    "Decoding audio and computing the full analysis."}
                  {state.phase === "ready" && "Recording ready."}
                  {state.phase === "idle" && "No active recording."}
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
                <CardTitle>Live monitor</CardTitle>
                <CardDescription>
                  The rolling window uses the highest peak seen in this recording session.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {state.phase === "starting" ? (
                  <PermissionPrompt />
                ) : !state.live || !liveAnalysis ? (
                  <>
                    <CanvasShell title="Waveform">
                      <EmptyChart label="No recording" detail="Start a recording to show the live signal." />
                    </CanvasShell>
                    <CanvasShell title="Spectrogram">
                      <EmptyChart
                        label="No recording"
                        detail="The live spectrogram appears while microphone capture is active."
                      />
                    </CanvasShell>
                  </>
                ) : (
                  <>
                    <div className="grid gap-4 md:grid-cols-2">
                      <CanvasShell title="Waveform">
                        <WaveformCanvas
                          samples={state.live.frame.samples}
                          sampleRate={state.live.frame.sampleRate}
                          label="Live waveform"
                          color="#00d4aa"
                          amplitudeReference={state.live.frame.peakAmplitude}
                          className="w-full"
                        />
                      </CanvasShell>

                      <div className="rounded-md border border-zinc-800 bg-zinc-950 p-4">
                        <dl>
                          <StatRow
                            label="Window"
                            value={`${(state.live.frame.elapsedMs / 1000).toFixed(1)} s`}
                          />
                          <StatRow
                            label="Input level"
                            value={`${(state.live.frame.level * 100).toFixed(1)} %`}
                          />
                          <StatRow
                            label="Session peak"
                            value={`${(state.live.frame.peakAmplitude * 100).toFixed(1)} %`}
                          />
                          <StatRow
                            label="Pitch"
                            value={
                              liveAnalysis.medianF0
                                ? `${liveAnalysis.medianF0.toFixed(1)} Hz`
                                : "Unvoiced"
                            }
                          />
                          <StatRow
                            label="Formants"
                            value={
                              liveAnalysis.formantMedians.length > 0
                                ? liveAnalysis.formantMedians
                                    .map((value) => Math.round(value))
                                    .join(" / ")
                                : "Tracking"
                            }
                          />
                        </dl>
                      </div>
                    </div>

                    <CanvasShell title="Spectrogram">
                      <SpectrogramCanvas
                        stft={liveAnalysis.stft}
                        pitchTrack={liveAnalysis.pitchTrack}
                        formants={liveAnalysis.formants}
                        maxFreqDisplay={Math.min(state.live.frame.sampleRate / 2, 8000)}
                        className="w-full"
                      />
                    </CanvasShell>
                  </>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Analysis</CardTitle>
                <CardDescription>
                  Final waveform, spectrogram, and edit comparison.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {busy ? (
                  <AnalysisSkeleton />
                ) : !ready ? (
                  <div className="space-y-5">
                    <CanvasShell title="Waveform">
                      <EmptyChart label="No recording" />
                    </CanvasShell>
                    <CanvasShell title="Spectrogram">
                      <EmptyChart label="No recording" />
                    </CanvasShell>
                  </div>
                ) : (
                  <Tabs defaultValue="original" className="w-full">
                    <TabsList>
                      <TabsTrigger value="original">Original</TabsTrigger>
                      <TabsTrigger value="edited">Edited</TabsTrigger>
                    </TabsList>

                    <TabsContent value="original" className="space-y-5">
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="rounded-md border border-zinc-800 bg-zinc-950 p-4">
                          <dl>
                            <StatRow
                              label="Duration"
                              value={`${(
                                state.recorded.samples.length / state.recorded.sampleRate
                              ).toFixed(2)} s`}
                            />
                            <StatRow
                              label="Frames"
                              value={String(state.analysis.stft.frames.length)}
                            />
                            <StatRow
                              label="Pitch"
                              value={
                                state.analysis.medianF0
                                  ? `${state.analysis.medianF0.toFixed(1)} Hz`
                                  : "Unvoiced"
                              }
                            />
                            <StatRow
                              label="Formants"
                              value={
                                state.analysis.formantMedians.length > 0
                                  ? state.analysis.formantMedians
                                      .map((value) => Math.round(value))
                                      .join(" / ")
                                  : "Unavailable"
                              }
                            />
                          </dl>
                        </div>

                        <CanvasShell title="Waveform">
                          <WaveformCanvas
                            samples={state.recorded.samples}
                            sampleRate={state.recorded.sampleRate}
                            label="Original waveform"
                            color="#00d4aa"
                            className="w-full"
                          />
                        </CanvasShell>
                      </div>

                      <CanvasShell title="Spectrogram">
                        <SpectrogramCanvas
                          stft={state.analysis.stft}
                          pitchTrack={state.analysis.pitchTrack}
                          formants={state.analysis.formants}
                          maxFreqDisplay={Math.min(state.recorded.sampleRate / 2, 8000)}
                          className="w-full"
                        />
                      </CanvasShell>
                    </TabsContent>

                    <TabsContent value="edited" className="space-y-5">
                      <div className="flex flex-wrap gap-2">
                        {presets.map((preset) => {
                          const active =
                            JSON.stringify(preset.edit) ===
                            JSON.stringify(state.selectedEdit);
                          return (
                            <Button
                              key={preset.label}
                              variant={active ? "secondary" : "outline"}
                              size="sm"
                              onClick={() => setSelectedEdit(preset.edit)}
                            >
                              {preset.label}
                            </Button>
                          );
                        })}
                      </div>

                      {state.applyingEdit || !state.edited ? (
                        <div className="space-y-4">
                          <SpinnerLabel text="Applying spectral edit." />
                          <AnalysisSkeleton />
                        </div>
                      ) : (
                        <>
                          <div className="grid gap-4 md:grid-cols-2">
                            <CanvasShell title="Edited waveform">
                              <WaveformCanvas
                                samples={state.edited.audio.samples}
                                sampleRate={state.edited.audio.sampleRate}
                                label="Edited waveform"
                                color="#ff9d5c"
                                className="w-full"
                              />
                            </CanvasShell>

                            <CanvasShell title="Difference">
                              <WaveformCanvas
                                samples={state.edited.difference}
                                sampleRate={state.recorded.sampleRate}
                                label="Original minus edited"
                                color="#d4d4d8"
                                className="w-full"
                              />
                            </CanvasShell>
                          </div>

                          <CanvasShell title="Edited spectrogram">
                            <SpectrogramCanvas
                              stft={state.edited.analysis.stft}
                              pitchTrack={state.edited.analysis.pitchTrack}
                              formants={state.edited.analysis.formants}
                              maxFreqDisplay={Math.min(state.recorded.sampleRate / 2, 8000)}
                              className="w-full"
                            />
                          </CanvasShell>
                        </>
                      )}
                    </TabsContent>
                  </Tabs>
                )}
              </CardContent>
            </Card>
          </div>

          <aside className="space-y-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Status</CardTitle>
              </CardHeader>
              <CardContent>
                <dl>
                  <StatRow label="Phase" value={state.phase} />
                  <StatRow label="Playback" value={state.playing ?? "idle"} />
                  <StatRow label="Edit" value={state.applyingEdit ? "updating" : "ready"} />
                </dl>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Current values</CardTitle>
              </CardHeader>
              <CardContent>
                {busy ? (
                  <div className="space-y-3">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-4 w-36" />
                    <Skeleton className="h-4 w-28" />
                  </div>
                ) : !ready ? (
                  <div className="rounded-md border border-dashed border-zinc-800 px-4 py-8 text-center text-sm text-zinc-400">
                    No recording
                  </div>
                ) : (
                  <dl>
                    <StatRow
                      label="Pitch"
                      value={
                        state.analysis.medianF0
                          ? `${state.analysis.medianF0.toFixed(1)} Hz`
                          : "Unvoiced"
                      }
                    />
                    <StatRow
                      label="Formants"
                      value={
                        state.analysis.formantMedians.length > 0
                          ? state.analysis.formantMedians
                              .map((value) => Math.round(value))
                              .join(" / ")
                          : "Unavailable"
                      }
                    />
                    <StatRow
                      label="Selected edit"
                      value={
                        presets.find(
                          (preset) =>
                            JSON.stringify(preset.edit) ===
                            JSON.stringify(state.selectedEdit),
                        )?.label ?? "Unknown"
                      }
                    />
                  </dl>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Processing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-zinc-400">
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "size-2 rounded-full bg-zinc-700",
                      state.phase === "recording" && "bg-emerald-500",
                    )}
                  />
                  Live monitor
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "size-2 rounded-full bg-zinc-700",
                      state.phase === "analyzing" && "bg-amber-500",
                    )}
                  />
                  Full analysis
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "size-2 rounded-full bg-zinc-700",
                      state.applyingEdit && "bg-amber-500",
                    )}
                  />
                  Spectral edit
                </div>
                {(state.microphonePermission === "denied" ||
                  state.microphonePermission === "unsupported") && (
                  <div className="flex items-start gap-2 border-t border-zinc-800 pt-3 text-amber-200">
                    <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                    <span>
                      Recording is currently unavailable until microphone access is restored.
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          </aside>
        </section>
      </div>
    </main>
  );
}
