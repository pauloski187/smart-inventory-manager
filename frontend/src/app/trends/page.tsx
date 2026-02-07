"use client";

import { useState, useEffect } from "react";
import { api, formatCurrency, formatPercent } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LineChart } from "@/components/charts/LineChart";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import type { MonthlySalesTrend } from "@/types/api";

export default function TrendsPage() {
  const [data, setData] = useState<MonthlySalesTrend | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [months, setMonths] = useState(12);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getMonthlySalesTrend(months);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [months]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  // Format data for chart
  const chartData = data.monthly_trends.map((trend) => ({
    month: new Date(trend.month + "-01").toLocaleDateString("en-US", {
      month: "short",
      year: "numeric",
    }),
    revenue: trend.revenue,
    profit: trend.profit,
  }));

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Sales Trends</h1>
          <p className="mt-2 text-slate-400">Revenue and profit analysis over time</p>
        </div>

        <select
          value={months}
          onChange={(e) => setMonths(Number(e.target.value))}
          className="rounded-lg border border-dark-border bg-dark-secondary text-white px-4 py-2 text-sm font-medium focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
        >
          <option value={3}>Last 3 months</option>
          <option value={6}>Last 6 months</option>
          <option value={12}>Last 12 months</option>
          <option value={24}>Last 24 months</option>
        </select>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Revenue</p>
            <h3 className="mt-2 text-2xl font-bold text-white">
              {formatCurrency(data.summary.total_revenue)}
            </h3>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Profit</p>
            <h3 className="mt-2 text-2xl font-bold text-accent-400">
              {formatCurrency(data.summary.total_profit)}
            </h3>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Avg Monthly Revenue</p>
            <h3 className="mt-2 text-2xl font-bold text-white">
              {formatCurrency(data.summary.avg_monthly_revenue)}
            </h3>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Avg Profit Margin</p>
            <h3 className="mt-2 text-2xl font-bold text-primary-400">
              {formatPercent(data.summary.avg_profit_margin)}
            </h3>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <Card>
        <CardHeader>
          <CardTitle>Revenue & Profit Trend</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={chartData}
            xKey="month"
            lines={[
              { key: "revenue", color: "#3B82F6", name: "Revenue" },
              { key: "profit", color: "#10B981", name: "Profit" },
            ]}
          />
        </CardContent>
      </Card>

      {/* Monthly Details Table */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Monthly Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="pb-3 font-semibold">Month</th>
                  <th className="pb-3 font-semibold text-right">Revenue</th>
                  <th className="pb-3 font-semibold text-right">Profit</th>
                  <th className="pb-3 font-semibold text-right">Margin</th>
                  <th className="pb-3 font-semibold text-right">Orders</th>
                  <th className="pb-3 font-semibold text-right">Growth</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {data.monthly_trends.map((trend, index) => (
                  <tr key={index} className="hover:bg-dark-hover">
                    <td className="py-3">{chartData[index].month}</td>
                    <td className="py-3 text-right font-medium">
                      {formatCurrency(trend.revenue)}
                    </td>
                    <td className="py-3 text-right font-medium text-accent-400">
                      {formatCurrency(trend.profit)}
                    </td>
                    <td className="py-3 text-right">{formatPercent(trend.profit_margin)}</td>
                    <td className="py-3 text-right">{trend.order_count.toLocaleString()}</td>
                    <td className={`py-3 text-right font-medium ${trend.revenue_growth_pct >= 0 ? "text-accent-400" : "text-danger-400"}`}>
                      {trend.revenue_growth_pct >= 0 ? "+" : ""}
                      {formatPercent(trend.revenue_growth_pct)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
