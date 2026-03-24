import type { AudioPreviewFrame, AudioSamples } from "../audio";
import {
  DEFAULT_SPECTRAL_EDIT,
  type EditedSignal,
  type SignalAnalysis,
  type SpectralEditKind,
} from "../speech-analysis";

export type MicrophonePermissionState =
  | "unknown"
  | "prompt"
  | "requesting"
  | "granted"
  | "denied"
  | "unsupported";

export type LiveAnalysis = {
  readonly frame: AudioPreviewFrame;
  readonly analysis: SignalAnalysis;
};

export type WorkbenchState = {
  readonly phase: "idle" | "starting" | "recording" | "analyzing" | "ready";
  readonly error: string | null;
  readonly applyingEdit: boolean;
  readonly microphonePermission: MicrophonePermissionState;
  readonly selectedEdit: SpectralEditKind;
  readonly editedFor: SpectralEditKind | null;
  readonly live: LiveAnalysis | null;
  readonly recorded: AudioSamples | null;
  readonly analysis: SignalAnalysis | null;
  readonly edited: EditedSignal | null;
};

export const initialWorkbenchState = (): WorkbenchState => ({
  phase: "idle",
  error: null,
  applyingEdit: false,
  microphonePermission: "unknown",
  selectedEdit: DEFAULT_SPECTRAL_EDIT,
  editedFor: null,
  live: null,
  recorded: null,
  analysis: null,
  edited: null,
});
