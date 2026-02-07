"use client";

import { useState, useEffect } from "react";
import { api, formatCurrency } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LineChart } from "@/components/charts/LineChart";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import type { SalesByDayOfWeek } from "@/types/api";

const DAY_COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#6366F1", "#EC4899", "#8B5CF6"];

export default function PatternsPage() {
  const [data, setData] = useState<SalesByDayOfWeek | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getSalesByDayOfWeek();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  const avgRevenue = data.daily_distribution.reduce((sum, day) => sum + day.revenue, 0) / data.daily_distribution.length;
  const bestDay = data.daily_distribution.reduce((prev, current) =>
    prev.revenue > current.revenue ? prev : current
  );
  const worstDay = data.daily_distribution.reduce((prev, current) =>
    prev.revenue < current.revenue ? prev : current
  );

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Sales Patterns</h1>
        <p className="mt-2 text-slate-400">Analyze sales performance by day of week</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Busiest Day</p>
            <h3 className="mt-2 text-2xl font-bold text-accent-400">{bestDay.day}</h3>
            <p className="mt-1 text-sm text-slate-500">{formatCurrency(bestDay.revenue)} revenue</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Slowest Day</p>
            <h3 className="mt-2 text-2xl font-bold text-danger-400">{worstDay.day}</h3>
            <p className="mt-1 text-sm text-slate-500">{formatCurrency(worstDay.revenue)} revenue</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Daily Average</p>
            <h3 className="mt-2 text-2xl font-bold text-primary-400">{formatCurrency(avgRevenue)}</h3>
            <p className="mt-1 text-sm text-slate-500">Across all days</p>
          </CardContent>
        </Card>
      </div>

      {/* Chart */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Revenue by Day of Week (Line Plot)</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={data.daily_distribution}
            xKey="day"
            lines={[
              { key: "revenue", color: "#3B82F6", name: "Revenue" }
            ]}
          />
        </CardContent>
      </Card>

      {/* Detailed Table */}
      <Card>
        <CardHeader>
          <CardTitle>Daily Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="pb-3 font-semibold">Day</th>
                  <th className="pb-3 text-right font-semibold">Revenue</th>
                  <th className="pb-3 text-right font-semibold">Profit</th>
                  <th className="pb-3 text-right font-semibold">Orders</th>
                  <th className="pb-3 text-right font-semibold">Avg Order Value</th>
                  <th className="pb-3 text-right font-semibold">vs Average</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {data.daily_distribution.map((day, index) => {
                  const vsAvg = ((day.revenue - avgRevenue) / avgRevenue * 100);
                  return (
                    <tr key={index} className="hover:bg-dark-hover">
                      <td className="py-3 font-medium">{day.day}</td>
                      <td className="py-3 text-right">{formatCurrency(day.revenue)}</td>
                      <td className="py-3 text-right text-accent-400">{formatCurrency(day.profit)}</td>
                      <td className="py-3 text-right">{day.order_count.toLocaleString()}</td>
                      <td className="py-3 text-right">{formatCurrency(day.avg_order_value)}</td>
                      <td className={`py-3 text-right font-medium ${vsAvg >= 0 ? "text-accent-400" : "text-danger-400"}`}>
                        {vsAvg >= 0 ? "+" : ""}{vsAvg.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Actionable Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-accent-500/10 rounded-lg">
              <div className="text-accent-400 mt-1">✓</div>
              <div>
                <p className="font-medium text-accent-300">Staffing Recommendation</p>
                <p className="text-sm text-accent-400 mt-1">
                  Schedule more staff on {bestDay.day} to handle {((bestDay.revenue / avgRevenue - 1) * 100).toFixed(0)}% higher volume
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-warning-500/10 rounded-lg">
              <div className="text-warning-400 mt-1">⚠</div>
              <div>
                <p className="font-medium text-warning-300">Promotion Opportunity</p>
                <p className="text-sm text-warning-400 mt-1">
                  Run special promotions on {worstDay.day} to boost sales during slower periods
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-primary-500/10 rounded-lg">
              <div className="text-primary-400 mt-1">ℹ</div>
              <div>
                <p className="font-medium text-primary-300">Inventory Planning</p>
                <p className="text-sm text-primary-400 mt-1">
                  Ensure higher stock levels for {bestDay.day} when demand is typically {((bestDay.revenue / avgRevenue - 1) * 100).toFixed(0)}% above average
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
