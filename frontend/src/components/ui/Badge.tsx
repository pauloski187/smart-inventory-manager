import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info";
  children: React.ReactNode;
}

export function Badge({ variant = "default", children, className, ...props }: BadgeProps) {
  const variantClasses = {
    default: "bg-dark-muted/50 text-slate-300 border-dark-border/30",
    success: "bg-accent-500/15 text-accent-400 border-accent-500/30",
    warning: "bg-warning-500/15 text-warning-400 border-warning-500/30",
    danger: "bg-danger-500/15 text-danger-400 border-danger-500/30",
    info: "bg-info-500/15 text-info-400 border-info-500/30",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold border",
        variantClasses[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
