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

export function AudioPlayerPanel({
  audioSrc,
}: {
  readonly audioSrc: string;
}) {
  return (
    <div className="flex h-[40px] items-center">
      <AudioPlayer className="w-full">
        <AudioPlayerElement key={audioSrc} src={audioSrc} preload="metadata" />
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
  );
}
