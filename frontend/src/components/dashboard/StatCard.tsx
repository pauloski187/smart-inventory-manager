import { Card, CardContent } from "@/components/ui/Card";
import { LucideIcon } from "lucide-react";
import { formatCurrency, formatNumber, formatPercent } from "@/lib/api";

interface StatCardProps {
  title: string;
  value: number;
  icon: LucideIcon;
  format?: "currency" | "number" | "percent";
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export function StatCard({ title, value, icon: Icon, format = "number", trend }: StatCardProps) {
  let formattedValue: string;

  switch (format) {
    case "currency":
      formattedValue = formatCurrency(value);
      break;
    case "percent":
      formattedValue = formatPercent(value);
      break;
    default:
      formattedValue = formatNumber(value);
  }

  return (
    <Card>
      <CardContent className="flex items-center justify-between pt-6">
        <div>
          <p className="text-sm font-medium text-slate-400">{title}</p>
          <h3 className="mt-2 text-3xl font-bold text-white">{formattedValue}</h3>
          {trend && (
            <p className={`mt-1 text-sm ${trend.isPositive ? "text-accent-400" : "text-danger-400"}`}>
              {trend.isPositive ? "+" : ""}{trend.value}%
            </p>
          )}
        </div>
        <div className="rounded-full bg-primary-500/15 p-3">
          <Icon className="h-6 w-6 text-primary-400" />
        </div>
      </CardContent>
    </Card>
  );
}
