import { AlertCircle } from "lucide-react";

interface ErrorMessageProps {
  message: string;
  title?: string;
}

export function ErrorMessage({ message, title = "Error" }: ErrorMessageProps) {
  return (
    <div className="rounded-xl border border-danger-500/30 bg-danger-500/10 p-6">
      <div className="flex items-start">
        <AlertCircle className="h-5 w-5 text-danger-400 mr-3 mt-0.5" />
        <div>
          <h3 className="text-sm font-medium text-danger-300">{title}</h3>
          <p className="mt-1 text-sm text-danger-400/80">{message}</p>
        </div>
      </div>
    </div>
  );
}
