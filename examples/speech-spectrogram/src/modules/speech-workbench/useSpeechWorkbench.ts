import {
  startTransition,
  useCallback,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import { Effect, Layer, ManagedRuntime } from "effect";

import { AudioInput, type AudioPreviewFrame, type RecordingSession } from "../audio";
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

const appLayer = Layer.mergeAll(AudioInput.layer, SpeechAnalysis.layer);
const appRuntime = ManagedRuntime.make(appLayer);
const MAX_RECORDING_MS = 60_000;

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
  "use no memo";

  const sessionRef = useRef<RecordingSession | null>(null);
  const autoStopTimeoutRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef<number | null>(null);
  const liveAnalysisTokenRef = useRef(0);
  const liveAnalysisInFlightRef = useRef(false);
  const pendingLiveFrameRef = useRef<AudioPreviewFrame | null>(null);
  const [state, setState] = useState<WorkbenchState>(() => initialWorkbenchState());
  const runtime = appRuntime;

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      startTransition(() => {
        setState((current) => ({ ...current, microphonePermission: "unsupported" }));
      });
      return;
    }

    if (!navigator.permissions?.query) {
      return;
    }

    let active = true;
    let permissionStatus: PermissionStatus | null = null;

    const syncPermission = (permissionState: PermissionState) => {
      if (!active) {
        return;
      }

      startTransition(() => {
        setState((current) => ({
          ...current,
          microphonePermission:
            permissionState === "granted"
              ? "granted"
              : permissionState === "denied"
                ? "denied"
                : "prompt",
        }));
      });
    };

    void navigator.permissions
      .query({ name: "microphone" as PermissionName })
      .then((status) => {
        permissionStatus = status;
        syncPermission(status.state);
        status.onchange = () => syncPermission(status.state);
      })
      .catch(() => undefined);

    return () => {
      active = false;
      if (permissionStatus) {
        permissionStatus.onchange = null;
      }
    };
  }, []);

  const handleError = useEffectEvent((error: unknown) => {
    const message = formatError(error);
    const permission = permissionFromError(error);
    startTransition(() => {
      setState((current) => ({
        ...current,
        phase: current.recorded ? "ready" : "idle",
        error: message,
        microphonePermission: permission ?? current.microphonePermission,
        autoStopNoticeOpen:
          current.autoStopNoticeOpen && current.phase !== "recording",
        applyingEdit: false,
      }));
    });
  });

  const analyzeLiveFrame = useEffectEvent((frame: AudioPreviewFrame) => {
    pendingLiveFrameRef.current = frame;
    if (liveAnalysisInFlightRef.current) {
      return;
    }

    const nextFrame = pendingLiveFrameRef.current;
    if (!nextFrame) {
      return;
    }

    pendingLiveFrameRef.current = null;
    liveAnalysisInFlightRef.current = true;
    const token = ++liveAnalysisTokenRef.current;

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const analysis = yield* SpeechAnalysis;
          const result = yield* analysis.analyzeSignal(
            { samples: nextFrame.samples, sampleRate: nextFrame.sampleRate },
            DEFAULT_ANALYSIS_CONFIG,
          );
          const live: LiveAnalysis = { frame: nextFrame, analysis: result };
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
      .catch(handleError)
      .finally(() => {
        liveAnalysisInFlightRef.current = false;
        if (pendingLiveFrameRef.current) {
          analyzeLiveFrame(pendingLiveFrameRef.current);
        }
      });
  });

  const startRecording = useCallback(() => {
    startTransition(() => {
      setState((current) => ({
        ...current,
        phase: "starting",
        error: null,
        applyingEdit: false,
        microphonePermission:
          current.microphonePermission === "prompt"
            ? "requesting"
            : current.microphonePermission,
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
            autoStopNoticeOpen: false,
            editedFor: null,
            live: null,
            recorded: null,
            analysis: null,
            edited: null,
          }));
        });
        recordingStartedAtRef.current = performance.now();
      })
      .catch(handleError);
  }, [analyzeLiveFrame, handleError, runtime]);

  const reset = useCallback(() => {
    const session = sessionRef.current;
    sessionRef.current = null;
    recordingStartedAtRef.current = null;

    if (session) {
      void runtime.runPromise(session.cancel).catch(handleError);
    }

    liveAnalysisTokenRef.current += 1;
    pendingLiveFrameRef.current = null;
    liveAnalysisInFlightRef.current = false;
    startTransition(() => {
      setState((current) => ({
        ...initialWorkbenchState(),
        microphonePermission: current.microphonePermission,
      }));
    });
  }, [handleError, runtime]);

  const stopRecording = useCallback((options?: { readonly automatic?: boolean }) => {
    const session = sessionRef.current;
    if (!session) {
      return;
    }

    sessionRef.current = null;
    recordingStartedAtRef.current = null;
    if (autoStopTimeoutRef.current != null) {
      window.clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
    liveAnalysisTokenRef.current += 1;
    startTransition(() => {
      setState((current) => ({
        ...current,
        phase: "analyzing",
        autoStopNoticeOpen: options?.automatic ?? false,
      }));
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
            editedFor: state.selectedEdit,
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
    setState((current) => {
      if (current.selectedEdit === edit) {
        return current;
      }

      return {
        ...current,
        selectedEdit: edit,
        applyingEdit: current.phase === "ready" && current.recorded !== null,
      };
    });
  }, []);

  const stopRecordingAutomatically = useEffectEvent(() => {
    stopRecording({ automatic: true });
  });

  useEffect(() => {
    if (state.phase !== "recording") {
      if (autoStopTimeoutRef.current != null) {
        window.clearTimeout(autoStopTimeoutRef.current);
        autoStopTimeoutRef.current = null;
      }
      return;
    }

    const startedAt = recordingStartedAtRef.current ?? performance.now();
    recordingStartedAtRef.current = startedAt;
    const remainingMs = Math.max(0, MAX_RECORDING_MS - (performance.now() - startedAt));

    autoStopTimeoutRef.current = window.setTimeout(() => {
      stopRecordingAutomatically();
    }, remainingMs);

    return () => {
      if (autoStopTimeoutRef.current != null) {
        window.clearTimeout(autoStopTimeoutRef.current);
        autoStopTimeoutRef.current = null;
      }
    };
  }, [state.phase, stopRecordingAutomatically]);

  useEffect(() => {
    if (state.phase !== "ready" || !state.recorded) {
      return;
    }
    if (state.editedFor === state.selectedEdit) {
      return;
    }

    const audio = state.recorded;
    const edit = state.selectedEdit;
    let cancelled = false;

    void runtime
      .runPromise(
        Effect.gen(function* () {
          const analysis = yield* SpeechAnalysis;
          return yield* analysis.applyEdit(audio, edit, DEFAULT_ANALYSIS_CONFIG);
        }),
      )
      .then((edited) => {
        if (cancelled) {
          return;
        }

        startTransition(() => {
          setState((current) => {
            if (current.recorded !== audio || current.selectedEdit !== edit) {
              return current;
            }
            return { ...current, edited, editedFor: edit, applyingEdit: false };
          });
        });
      })
      .catch((error) => {
        if (!cancelled) {
          handleError(error);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [handleError, runtime, state.editedFor, state.phase, state.recorded, state.selectedEdit]);

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
      if (autoStopTimeoutRef.current != null) {
        window.clearTimeout(autoStopTimeoutRef.current);
        autoStopTimeoutRef.current = null;
      }
      recordingStartedAtRef.current = null;
      const session = sessionRef.current;
      sessionRef.current = null;

      if (session) {
        void runtime.runPromise(session.cancel).catch(() => undefined);
      }
    };
  }, [runtime]);

  const dismissAutoStopNotice = useCallback(() => {
    startTransition(() => {
      setState((current) => ({ ...current, autoStopNoticeOpen: false }));
    });
  }, []);

  return {
    state,
    presets: SPECTRAL_EDIT_PRESETS,
    startRecording,
    stopRecording,
    reset,
    setSelectedEdit,
    dismissAutoStopNotice,
  };
}
