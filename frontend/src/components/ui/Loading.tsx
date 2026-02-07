export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-500 border-t-transparent"></div>
    </div>
  );
}

export function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 w-3/4 rounded-lg bg-dark-secondary"></div>
      <div className="h-32 w-full rounded-lg bg-dark-secondary"></div>
      <div className="grid grid-cols-3 gap-4">
        <div className="h-24 rounded-lg bg-dark-secondary"></div>
        <div className="h-24 rounded-lg bg-dark-secondary"></div>
        <div className="h-24 rounded-lg bg-dark-secondary"></div>
      </div>
    </div>
  );
}
