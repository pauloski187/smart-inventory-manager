"use client";

import { useState, useEffect } from "react";
import { api, formatCurrency, formatPercent } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LineChart } from "@/components/charts/LineChart";
import { PieChart } from "@/components/charts/PieChart";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import type { AvailableMonths, MonthlyReport } from "@/types/api";

interface Insight {
  type: "info" | "success" | "warning" | "critical";
  title: string;
  message: string;
}

function generateInsights(report: MonthlyReport): Insight[] {
  const insights: Insight[] = [];
  const { summary, top_products, worst_products, category_breakdown } = report;

  // 1. Revenue & Profit Assessment
  if (summary.profit_margin >= 35) {
    insights.push({
      type: "success",
      title: "Excellent Profit Margin",
      message: `Profit margin of ${summary.profit_margin.toFixed(1)}% is well above the 30% benchmark. Total profit of $${summary.total_profit.toLocaleString(undefined, { maximumFractionDigits: 0 })} shows strong pricing strategy and cost control.`,
    });
  } else if (summary.profit_margin >= 20) {
    insights.push({
      type: "info",
      title: "Healthy Profit Margin",
      message: `Profit margin at ${summary.profit_margin.toFixed(1)}% is within healthy range. Consider reviewing high-cost products to push margin above 30%.`,
    });
  } else {
    insights.push({
      type: "warning",
      title: "Low Profit Margin",
      message: `Profit margin of ${summary.profit_margin.toFixed(1)}% is below the 20% threshold. Review supplier costs and pricing strategy to improve profitability.`,
    });
  }

  // 2. Top Category Dominance
  if (category_breakdown.length > 0) {
    const topCategory = [...category_breakdown].sort((a, b) => b.revenue - a.revenue)[0];
    const bottomCategory = [...category_breakdown].sort((a, b) => a.revenue - b.revenue)[0];

    if (topCategory.revenue_share > 30) {
      insights.push({
        type: "warning",
        title: "Revenue Concentration Risk",
        message: `${topCategory.category} accounts for ${topCategory.revenue_share.toFixed(1)}% of revenue. High dependency on a single category increases business risk. Consider diversifying into other categories.`,
      });
    } else {
      insights.push({
        type: "success",
        title: "Balanced Revenue Distribution",
        message: `Revenue is well-distributed across categories. ${topCategory.category} leads at ${topCategory.revenue_share.toFixed(1)}% while ${bottomCategory.category} contributes ${bottomCategory.revenue_share.toFixed(1)}%. Good diversification reduces risk.`,
      });
    }
  }

  // 3. Top Product Spotlight
  if (top_products.length > 0) {
    const star = top_products[0];
    insights.push({
      type: "success",
      title: "Star Product",
      message: `"${star.product_name}" (${star.category}) is the top performer with $${star.revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })} revenue from ${star.units_sold} units. Consider increasing stock and marketing spend for this product.`,
    });
  }

  // 4. Underperformer Alert
  if (worst_products.length > 0) {
    const weak = worst_products[0];
    insights.push({
      type: "critical",
      title: "Underperformer Alert",
      message: `"${weak.product_name}" (${weak.category}) generated only $${weak.revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })} from ${weak.units_sold} units. Evaluate whether to discount, rebundle, or discontinue this product.`,
    });
  }

  // 5. Order Volume & Efficiency
  const avgUnitsPerOrder = summary.total_units_sold / summary.total_orders;
  insights.push({
    type: "info",
    title: "Order Efficiency",
    message: `${summary.total_orders} orders processed averaging $${summary.avg_order_value.toFixed(2)} per order and ${avgUnitsPerOrder.toFixed(1)} units per order. ${
      summary.avg_order_value > 100
        ? "Strong average order value indicates good upselling."
        : "Consider bundling products or minimum order incentives to increase order value."
    }`,
  });

  // 6. Loss Analysis
  if (summary.total_loss > 0) {
    const lossRatio = (summary.total_loss / summary.total_revenue) * 100;
    insights.push({
      type: lossRatio > 5 ? "critical" : "warning",
      title: "Loss Detected",
      message: `Total losses of $${summary.total_loss.toLocaleString(undefined, { maximumFractionDigits: 0 })} recorded (${lossRatio.toFixed(1)}% of revenue). Investigate return rates, damaged goods, and pricing errors to reduce losses.`,
    });
  }

  return insights;
}

export default function ReportsPage() {
  const [availableMonths, setAvailableMonths] = useState<AvailableMonths | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<{ year: number; month: number } | null>(null);
  const [report, setReport] = useState<MonthlyReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAvailableMonths() {
      try {
        setLoading(true);
        setError(null);
        const months = await api.getAvailableMonths();
        setAvailableMonths(months);

        // Auto-select the most recent month
        if (months.available_months.length > 0) {
          const latest = months.available_months[0];
          setSelectedMonth({ year: latest.year, month: latest.month });
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load available months");
      } finally {
        setLoading(false);
      }
    }

    fetchAvailableMonths();
  }, []);

  useEffect(() => {
    async function fetchReport() {
      if (!selectedMonth) return;

      try {
        setLoading(true);
        setError(null);
        const reportData = await api.getMonthlyReport(selectedMonth.year, selectedMonth.month);
        setReport(reportData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load report");
      } finally {
        setLoading(false);
      }
    }

    fetchReport();
  }, [selectedMonth]);

  if (loading && !report) return <LoadingSpinner />;
  if (error && !availableMonths) return <ErrorMessage message={error} />;

  const categoryPieData = report?.category_breakdown.map(cat => ({
    name: cat.category,
    value: cat.revenue_share,
  })) || [];

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">Monthly Reports</h1>
            <p className="mt-2 text-slate-400">Detailed monthly performance analysis</p>
          </div>

          {/* Month Selector */}
          {availableMonths && availableMonths.available_months.length > 0 && (
            <select
              value={selectedMonth ? `${selectedMonth.year}-${selectedMonth.month}` : ""}
              onChange={(e) => {
                const [year, month] = e.target.value.split("-").map(Number);
                setSelectedMonth({ year, month });
              }}
              className="px-4 py-2 border border-dark-border rounded-lg shadow-sm bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
            >
              {availableMonths.available_months.map((m) => (
                <option key={`${m.year}-${m.month}`} value={`${m.year}-${m.month}`}>
                  {m.display} ({m.order_count} orders)
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {!report ? (
        <p className="text-slate-500">Select a month to view report</p>
      ) : (
        <>
          {/* Report Header */}
          <Card className="mb-6 bg-gradient-to-r from-primary-500/10 to-primary-500/5 border-l-4 border-l-blue-500">
            <CardContent className="pt-6">
              <h2 className="text-2xl font-bold text-white">
                {report.report_period.month_name} {report.report_period.year}
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                {report.report_period.start_date} to {report.report_period.end_date}
              </p>
            </CardContent>
          </Card>

          {/* Summary Cards */}
          <div className="grid gap-6 md:grid-cols-4 mb-8">
            <Card>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Total Revenue</p>
                <h3 className="mt-2 text-2xl font-bold text-primary-400">{formatCurrency(report.summary.total_revenue)}</h3>
                <p className="mt-1 text-sm text-slate-500">{report.summary.total_orders} orders</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Total Profit</p>
                <h3 className="mt-2 text-2xl font-bold text-accent-400">{formatCurrency(report.summary.total_profit)}</h3>
                <p className="mt-1 text-sm text-slate-500">{formatPercent(report.summary.profit_margin)} margin</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Avg Order Value</p>
                <h3 className="mt-2 text-2xl font-bold text-primary-400">{formatCurrency(report.summary.avg_order_value)}</h3>
                <p className="mt-1 text-sm text-slate-500">{report.summary.total_units_sold} units sold</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Total Loss</p>
                <h3 className="mt-2 text-2xl font-bold text-danger-400">{formatCurrency(report.summary.total_loss)}</h3>
                <p className="mt-1 text-sm text-slate-500">Losses recorded</p>
              </CardContent>
            </Card>
          </div>

          {/* Daily Breakdown Chart */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Daily Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={report.daily_breakdown}
                xKey="date"
                lines={[
                  { key: "revenue", color: "#3B82F6", name: "Revenue" },
                  { key: "profit", color: "#10B981", name: "Profit" },
                ]}
              />
            </CardContent>
          </Card>

          {/* Charts Row */}
          <div className="grid gap-6 lg:grid-cols-2 mb-6">
            {/* Category Breakdown Pie Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Revenue by Category</CardTitle>
              </CardHeader>
              <CardContent>
                <PieChart data={categoryPieData} />
              </CardContent>
            </Card>

            {/* Category Table */}
            <Card>
              <CardHeader>
                <CardTitle>Category Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="border-b border-dark-border">
                      <tr className="text-left">
                        <th className="pb-3 font-semibold">Category</th>
                        <th className="pb-3 text-right font-semibold">Revenue</th>
                        <th className="pb-3 text-right font-semibold">Profit</th>
                        <th className="pb-3 text-right font-semibold">Share</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-dark-border">
                      {report.category_breakdown.map((cat, index) => (
                        <tr key={index} className="hover:bg-dark-hover">
                          <td className="py-3 font-medium">{cat.category}</td>
                          <td className="py-3 text-right">{formatCurrency(cat.revenue)}</td>
                          <td className="py-3 text-right text-accent-400">{formatCurrency(cat.profit)}</td>
                          <td className="py-3 text-right">{formatPercent(cat.revenue_share)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Products Performance */}
          <div className="grid gap-6 lg:grid-cols-2 mb-6">
            {/* Top Products */}
            <Card>
              <CardHeader>
                <CardTitle>Top Performing Products</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {report.top_products.map((product, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-accent-500/10 rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium text-white">{product.product_name}</p>
                        <p className="text-sm text-slate-400">{product.category}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-semibold text-accent-400">{formatCurrency(product.revenue)}</p>
                        <p className="text-sm text-slate-400">{product.units_sold} units</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Worst Products */}
            <Card>
              <CardHeader>
                <CardTitle>Underperforming Products</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {report.worst_products.map((product, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-danger-500/10 rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium text-white">{product.product_name}</p>
                        <p className="text-sm text-slate-400">{product.category}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-semibold text-danger-400">{formatCurrency(product.revenue)}</p>
                        <p className="text-sm text-slate-400">{product.units_sold} units</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Insights & Recommendations - Generated from report data */}
          <Card>
            <CardHeader>
              <CardTitle>Insights & Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {generateInsights(report).map((insight, index) => {
                  const colors = {
                    info: { bg: "bg-primary-500/10", border: "border-primary-500/30", title: "text-primary-300", text: "text-primary-400", icon: "ℹ️", iconColor: "text-primary-400" },
                    success: { bg: "bg-accent-500/10", border: "border-accent-500/30", title: "text-accent-300", text: "text-accent-400", icon: "✅", iconColor: "text-accent-400" },
                    warning: { bg: "bg-warning-500/10", border: "border-warning-500/30", title: "text-warning-300", text: "text-warning-400", icon: "⚠️", iconColor: "text-warning-400" },
                    critical: { bg: "bg-danger-500/10", border: "border-red-200", title: "text-danger-300", text: "text-danger-400", icon: "🔴", iconColor: "text-danger-400" },
                  };
                  const style = colors[insight.type];

                  return (
                    <div key={index} className={`flex items-start gap-3 p-4 ${style.bg} rounded-lg border ${style.border}`}>
                      <div className={`${style.iconColor} mt-0.5 text-lg flex-shrink-0`}>{style.icon}</div>
                      <div>
                        <p className={`text-sm ${style.title} font-semibold`}>{insight.title}</p>
                        <p className={`text-sm ${style.text} mt-1`}>{insight.message}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
