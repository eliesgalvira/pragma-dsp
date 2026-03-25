import type * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "../../lib/utils";

const alertVariants = cva(
  "relative w-full rounded-md border px-4 py-3 text-sm",
  {
    variants: {
      variant: {
        default: "border-zinc-800 bg-zinc-900 text-zinc-100",
        warning: "border-amber-900 bg-amber-950/40 text-amber-100",
        destructive: "border-rose-900 bg-rose-950/40 text-rose-200",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const Alert = ({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>) => (
  <div role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
);

const AlertTitle = ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h5 className={cn("mb-1 font-medium", className)} {...props} />
);

const AlertDescription = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("text-sm text-inherit/90", className)} {...props} />
);

export { Alert, AlertDescription, AlertTitle };
