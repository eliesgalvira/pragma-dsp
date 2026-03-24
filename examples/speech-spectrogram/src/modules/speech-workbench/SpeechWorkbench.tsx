import type * as React from "react";
import {
  Mic,
  RotateCcw,
  Square,
  Volume2,
  WandSparkles,
} from "lucide-react";

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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import { cn } from "../../lib/utils";
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

function AnalysisSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-[180px] w-full" />
        <Skeleton className="h-[180px] w-full" />
      </div>
      <Skeleton className="h-[320px] w-full" />
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
  const waitingForPermission =
    state.phase === "starting" && state.microphonePermission === "requesting";
  const liveEmptyLabel =
    state.phase === "recording" ? "Recording is in progress" : "No active recording";
  const liveEmptyDetail =
    state.phase === "recording"
      ? "Waiting for the first microphone frames to populate the live monitor."
      : "Start a recording to show the live signal and spectrogram.";
  const analysisEmptyLabel =
    state.phase === "recording" ? "Recording is in progress" : "No recording";
  const analysisEmptyDetail =
    state.phase === "recording"
      ? "Final waveform and spectrogram appear after you stop recording."
      : "Record audio to populate the analysis panels.";
  const showPrimaryRecordButton =
    state.phase === "idle" ||
    state.phase === "ready" ||
    state.phase === "starting" ||
    state.phase === "analyzing";

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

                      <Button
                        variant="outline"
                        onClick={playOriginal}
                        disabled={!ready || state.playing !== null}
                      >
                        {state.playing === "original" ? (
                          <InlineSpinner />
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
                          <InlineSpinner />
                        ) : (
                          <WandSparkles className="size-4" />
                        )}
                        Play edited
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
                <CardTitle>Live monitor</CardTitle>
                <CardDescription>
                  The rolling window uses the highest peak seen in this recording session.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {!state.live || !liveAnalysis ? (
                  <>
                    <div className="grid gap-4 md:grid-cols-2">
                      <CanvasShell title="Waveform">
                        <EmptyChart
                          label={liveEmptyLabel}
                          detail={liveEmptyDetail}
                          height={180}
                        />
                      </CanvasShell>

                      <PanelShell title="Session values">
                        <FixedStatPanel>
                          <div className="flex h-full items-center justify-center text-center">
                            <div className="space-y-1">
                              <p className="text-sm font-medium text-zinc-200">
                                {liveEmptyLabel}
                              </p>
                              <p className="text-sm text-zinc-500">
                                Session values appear after the live stream starts.
                              </p>
                            </div>
                          </div>
                        </FixedStatPanel>
                      </PanelShell>
                    </div>

                    <CanvasShell title="Spectrogram">
                      <EmptyChart
                        label={liveEmptyLabel}
                        detail="The live spectrogram appears while microphone capture is active."
                        height={320}
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

                      <PanelShell title="Session values">
                        <FixedStatPanel>
                          <dl className="grid h-full grid-rows-5">
                            <StatRow
                              label="Window"
                              value={`${(state.live.frame.elapsedMs / 1000).toFixed(1)} s`}
                              className="h-full py-0"
                            />
                            <StatRow
                              label="Input level"
                              value={`${(state.live.frame.level * 100).toFixed(1)} %`}
                              className="h-full py-0"
                            />
                            <StatRow
                              label="Session peak"
                              value={`${(state.live.frame.peakAmplitude * 100).toFixed(1)} %`}
                              className="h-full py-0"
                            />
                            <StatRow
                              label="Pitch"
                              value={
                                liveAnalysis.medianF0
                                  ? `${liveAnalysis.medianF0.toFixed(1)} Hz`
                                  : "Unvoiced"
                              }
                              className="h-full py-0"
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
                              className="h-full py-0"
                            />
                          </dl>
                        </FixedStatPanel>
                      </PanelShell>
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
                      <EmptyChart
                        label={analysisEmptyLabel}
                        detail={analysisEmptyDetail}
                        height={180}
                      />
                    </CanvasShell>
                    <CanvasShell title="Spectrogram">
                      <EmptyChart
                        label={analysisEmptyLabel}
                        detail={analysisEmptyDetail}
                        height={320}
                      />
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
                        <PanelShell title="Waveform values">
                          <FixedStatPanel>
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
                          </FixedStatPanel>
                        </PanelShell>

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
        </section>
      </div>
    </main>
  );
}
