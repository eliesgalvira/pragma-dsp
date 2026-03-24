import * as React from "react";

import { cn } from "@/lib/utils";

type SliderProps = Omit<React.ComponentProps<"input">, "type" | "value" | "onChange"> & {
  readonly value: number;
  readonly onValueChange: (value: number) => void;
};

function Slider({ className, value, onValueChange, ...props }: SliderProps) {
  return (
    <input
      type="range"
      value={value}
      className={cn(
        "h-2 w-full cursor-pointer appearance-none rounded-full bg-zinc-800 accent-zinc-100",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400/50 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      onChange={(event) => onValueChange(Number(event.target.value))}
      {...props}
    />
  );
}

export { Slider };
