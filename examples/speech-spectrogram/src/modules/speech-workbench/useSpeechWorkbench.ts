import {
  startTransition,
  useCallback,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import { Effect, Layer, ManagedRuntime } from "effect";

import { AudioInput, AudioOutput, type AudioPreviewFrame, type AudioSamples, type RecordingSession } from "../audio";
import {
  DEFAULT_ANALYSIS_CONFIG,
  SPECTRAL_EDIT_PRESETS,
  SpeechAnalysis,
  type SpectralEditKind,
} from "../speech-analysis";
import {
  initialWorkbenchState,
  type LiveAnalysis,
  type MicrophonePermissionState,
  type WorkbenchState,
} from "./model";

const appLayer = Layer.mergeAll(AudioInput.layer, AudioOutput.layer, SpeechAnalysis.layer);
const appRuntime = ManagedRuntime.make(appLayer);

const formatError = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const permissionFromError = (error: unknown): MicrophonePermissionState | null => {
  if (!(error instanceof Error) || !("_tag" in error)) {
    return null;
  }

  if ("code" in error) {
    const code = error.code;
    if (code === "permission-denied") return "denied";
    if (code === "unsupported-browser") return "unsupported";
  }

  return null;
};

export function useSpeechWorkbench() {
  const sessionRef = useRef<RecordingSession | null>(null);
  const liveAnalysisTokenRef = useRef(0);
  const [state, setState] = useState<WorkbenchState>(() => initialWorkbenchState());
  const runtime = appRuntime;

  const handleError = useEffectEvent((error: unknown) => {
    const message = formatError(error);
    const permission = permissionFromError(error);
    startTransition(() => {
      setState((current) => ({
        ...current,
        phase: current.recorded ? "ready" : "idle",
        error: message,
        microphonePermission: permission ?? current.microphonePermission,
        playing: null,
        applyingEdit: false,
      }));
    });
  });

  const applyEdit = useEffectEvent((audio: AudioSamples, edit: SpectralEditKind) => {
    startTransition(() => {
      setState((current) => {
        if (current.recorded !== audio) {
          return current;
        }
        return { ...current, applyingEdit: true };
      });
    });

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const analysis = yield* SpeechAnalysis;
          return yield* analysis.applyEdit(audio, edit, DEFAULT_ANALYSIS_CONFIG);
        }),
      )
      .then((edited) => {
        startTransition(() => {
          setState((current) => {
            if (current.recorded !== audio) {
              return current;
            }
            return { ...current, edited, applyingEdit: false };
          });
        });
      })
      .catch(handleError);
  });

  const analyzeLiveFrame = useEffectEvent((frame: AudioPreviewFrame) => {
    const token = ++liveAnalysisTokenRef.current;

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const analysis = yield* SpeechAnalysis;
          const result = yield* analysis.analyzeSignal(
            { samples: frame.samples, sampleRate: frame.sampleRate },
            DEFAULT_ANALYSIS_CONFIG,
          );
          const live: LiveAnalysis = { frame, analysis: result };
          return live;
        }),
      )
      .then((live) => {
        if (token !== liveAnalysisTokenRef.current) {
          return;
        }

        startTransition(() => {
          setState((current) => {
            if (current.phase !== "recording") {
              return current;
            }
            return { ...current, live };
          });
        });
      })
      .catch(handleError);
  });

  const startRecording = useCallback(() => {
    startTransition(() => {
      setState((current) => ({
        ...current,
        phase: "starting",
        error: null,
        playing: null,
        applyingEdit: false,
        microphonePermission:
          current.microphonePermission === "granted" ? "granted" : "requesting",
      }));
    });

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const audioInput = yield* AudioInput;
          return yield* audioInput.startRecording({
            sampleRate: 16_000,
            previewFftSize: DEFAULT_ANALYSIS_CONFIG.previewFftSize,
            previewWindowMs: DEFAULT_ANALYSIS_CONFIG.previewWindowMs,
            previewIntervalMs: DEFAULT_ANALYSIS_CONFIG.previewIntervalMs,
            onFrame: analyzeLiveFrame,
          });
        }),
      )
      .then((session) => {
        sessionRef.current = session;
        startTransition(() => {
          setState((current) => ({
            ...current,
            phase: "recording",
            microphonePermission: "granted",
            live: null,
            recorded: null,
            analysis: null,
            edited: null,
          }));
        });
      })
      .catch(handleError);
  }, [analyzeLiveFrame, handleError, runtime]);

  const reset = useCallback(() => {
    const session = sessionRef.current;
    sessionRef.current = null;

    if (session) {
      void runtime.runPromise(session.cancel).catch(handleError);
    }

    liveAnalysisTokenRef.current += 1;
    startTransition(() => {
      setState(initialWorkbenchState());
    });
  }, [handleError, runtime]);

  const stopRecording = useCallback(() => {
    const session = sessionRef.current;
    if (!session) {
      return;
    }

    sessionRef.current = null;
    liveAnalysisTokenRef.current += 1;
    startTransition(() => {
      setState((current) => ({ ...current, phase: "analyzing" }));
    });

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const audio = yield* session.stop;
          const analysis = yield* SpeechAnalysis;
          const result = yield* analysis.analyzeSignal(audio, DEFAULT_ANALYSIS_CONFIG);
          const edited = yield* analysis.applyEdit(audio, state.selectedEdit, DEFAULT_ANALYSIS_CONFIG);
          return { audio, analysis: result, edited };
        }),
      )
      .then(({ audio, analysis, edited }) => {
        startTransition(() => {
          setState((current) => ({
            ...current,
            phase: "ready",
            applyingEdit: false,
            microphonePermission: "granted",
            live: null,
            recorded: audio,
            analysis,
            edited,
          }));
        });
      })
      .catch(handleError);
  }, [handleError, runtime, state.selectedEdit]);

  const setSelectedEdit = useCallback((edit: SpectralEditKind) => {
    startTransition(() => {
      setState((current) => ({ ...current, selectedEdit: edit }));
    });
  }, []);

  useEffect(() => {
    if (state.phase !== "ready" || !state.recorded) {
      return;
    }

    applyEdit(state.recorded, state.selectedEdit);
  }, [applyEdit, state.phase, state.recorded, state.selectedEdit]);

  const play = useEffectEvent((which: "original" | "edited") => {
    const audio = which === "original" ? state.recorded : state.edited?.audio;
    if (!audio) {
      return;
    }

    startTransition(() => {
      setState((current) => ({ ...current, playing: which, error: null }));
    });

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const output = yield* AudioOutput;
          yield* output.play(audio);
        }),
      )
      .then(() => {
        startTransition(() => {
          setState((current) => ({ ...current, playing: null }));
        });
      })
      .catch(handleError);
  });

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      startTransition(() => {
        setState((current) => ({ ...current, microphonePermission: "unsupported" }));
      });
      return;
    }

    if (!("permissions" in navigator) || typeof navigator.permissions?.query !== "function") {
      startTransition(() => {
        setState((current) => ({
          ...current,
          microphonePermission:
            current.microphonePermission === "unknown"
              ? "prompt"
              : current.microphonePermission,
        }));
      });
      return;
    }

    let cancelled = false;
    void navigator.permissions
      .query({ name: "microphone" as PermissionName })
      .then((status) => {
        if (cancelled) {
          return;
        }

        const readState = () =>
          status.state === "granted"
            ? "granted"
            : status.state === "denied"
              ? "denied"
              : "prompt";

        startTransition(() => {
          setState((current) => ({ ...current, microphonePermission: readState() }));
        });

        status.onchange = () => {
          startTransition(() => {
            setState((current) => ({ ...current, microphonePermission: readState() }));
          });
        };
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setState((current) => ({
            ...current,
            microphonePermission:
              current.microphonePermission === "unknown"
                ? "prompt"
                : current.microphonePermission,
          }));
        });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      const session = sessionRef.current;
      sessionRef.current = null;

      if (session) {
        void runtime.runPromise(session.cancel).catch(() => undefined);
      }
    };
  }, [runtime]);

  return {
    state,
    presets: SPECTRAL_EDIT_PRESETS,
    startRecording,
    stopRecording,
    reset,
    setSelectedEdit,
    playOriginal: () => play("original"),
    playEdited: () => play("edited"),
  };
}
