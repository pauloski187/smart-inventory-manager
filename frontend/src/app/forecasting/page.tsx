"use client";

import { useState, useEffect } from "react";
import { api, formatNumber } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LineChart } from "@/components/charts/LineChart";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";

interface CategoryForecast {
  category: string;
  forecast_period_days: number;
  daily_forecast?: Array<{ date: string; forecast: number }>;
  forecast_90_day?: number;
  forecast_total?: number;
  confidence_interval?: { lower: number; upper: number };
  daily_average?: number;
  recommended_reorder_point?: number;
  recommended_safety_stock?: number;
  stockout_risk?: string;
}

// Helper function to safely get a number value
function safeNumber(value: any): number {
  const num = Number(value);
  return isNaN(num) || !isFinite(num) ? 0 : num;
}

export default function ForecastingPage() {
  const [availableCategories, setAvailableCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [forecastDays, setForecastDays] = useState<number>(90);
  const [forecast, setForecast] = useState<CategoryForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingCategories, setLoadingCategories] = useState(true);

  // Fetch available categories on mount
  useEffect(() => {
    async function fetchCategories() {
      try {
        setLoadingCategories(true);
        const categoryData = await api.getCategoryPerformance();
        const cats = categoryData.categories.map(c => c.category);
        setAvailableCategories(cats);
        if (cats.length > 0) {
          setSelectedCategory(cats[0]); // Select first category by default
        }
      } catch (err) {
        console.error("Error fetching categories:", err);
        setError("Failed to load categories");
      } finally {
        setLoadingCategories(false);
      }
    }

    fetchCategories();
  }, []);

  // Fetch forecast when category or days change
  useEffect(() => {
    if (!selectedCategory) return;

    async function fetchForecast() {
      try {
        setLoading(true);
        setError(null);

        const data = await api.getCategoryForecast(selectedCategory, true);
        setForecast(data);
      } catch (err) {
        console.error("Forecasting error:", err);
        setError(err instanceof Error ? err.message : "Failed to load forecast");
        setForecast(null);
      } finally {
        setLoading(false);
      }
    }

    fetchForecast();
  }, [selectedCategory, forecastDays]);

  if (loadingCategories) {
    return <LoadingSpinner />;
  }

  if (availableCategories.length === 0) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Demand Forecasting</h1>
          <p className="mt-2 text-slate-400">AI-powered demand predictions and inventory recommendations</p>
        </div>
        <Card className="border-l-4 border-l-warning-500">
          <CardContent className="pt-8 pb-8">
            <div className="text-center">
              <div className="text-4xl mb-4">⚠️</div>
              <h2 className="text-xl font-bold text-white mb-2">No Category Data Available</h2>
              <p className="text-slate-400">
                Please ensure your database has sales data with product categories.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Prepare chart data from daily forecasts
  const chartData = forecast?.daily_forecast
    ?.slice(0, forecastDays)
    .map((d, idx) => ({
      day: `Day ${idx + 1}`,
      forecast: safeNumber(d.forecast),
    })) || [];

  const totalForecast = safeNumber(forecast?.forecast_total || forecast?.forecast_90_day);
  const dailyAvg = safeNumber(forecast?.daily_average);
  const reorderPoint = safeNumber(forecast?.recommended_reorder_point);
  const safetyStock = safeNumber(forecast?.recommended_safety_stock);
  const confidenceLower = safeNumber(forecast?.confidence_interval?.lower);
  const confidenceUpper = safeNumber(forecast?.confidence_interval?.upper);
  const stockoutRisk = forecast?.stockout_risk || "low";

  return (
    <div>
      {/* Header with Controls */}
      <div className="mb-8">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white">Demand Forecasting</h1>
            <p className="mt-2 text-slate-400">AI-powered demand predictions and inventory recommendations</p>
          </div>

          {/* Control Panel */}
          <div className="flex items-center gap-4">
            {/* Category Selector */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Select Category
              </label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-4 py-2.5 border border-dark-border rounded-lg shadow-sm bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400 min-w-[200px]"
              >
                {availableCategories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>

            {/* Days Input */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Forecast Days (1-90)
              </label>
              <input
                type="number"
                min="1"
                max="90"
                value={forecastDays}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  if (val >= 1 && val <= 90) {
                    setForecastDays(val);
                  }
                }}
                className="px-4 py-2.5 border border-dark-border rounded-lg shadow-sm bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400 w-[120px]"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center items-center py-20">
          <LoadingSpinner />
        </div>
      )}

      {/* Error State */}
      {error && !loading && <ErrorMessage message={error} />}

      {/* Forecast Display */}
      {!loading && !error && forecast && (
        <>
          {/* Summary Cards */}
          <div className="grid gap-6 md:grid-cols-4 mb-8">
            <Card className="border-l-4 border-l-primary">
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Total Forecast</p>
                <h3 className="mt-2 text-3xl font-bold text-primary-400">{formatNumber(totalForecast)}</h3>
                <p className="mt-1 text-sm text-slate-500">{forecastDays} days ahead</p>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-blue-500">
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Daily Average</p>
                <h3 className="mt-2 text-3xl font-bold text-primary-400">{formatNumber(dailyAvg)}</h3>
                <p className="mt-1 text-sm text-slate-500">units per day</p>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-yellow-500">
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Reorder Point</p>
                <h3 className="mt-2 text-3xl font-bold text-warning-400">{formatNumber(reorderPoint)}</h3>
                <p className="mt-1 text-sm text-slate-500">units</p>
              </CardContent>
            </Card>

            <Card className={`border-l-4 ${
              stockoutRisk === "high" ? "border-l-red-500" :
              stockoutRisk === "medium" ? "border-l-yellow-500" : "border-l-green-500"
            }`}>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-slate-400">Stockout Risk</p>
                <div className="mt-2">
                  <Badge
                    variant={
                      stockoutRisk === "high" ? "danger" :
                      stockoutRisk === "medium" ? "warning" : "success"
                    }
                    className="text-lg px-4 py-1"
                  >
                    {stockoutRisk.toUpperCase()}
                  </Badge>
                </div>
                <p className="mt-2 text-sm text-slate-500">Risk assessment</p>
              </CardContent>
            </Card>
          </div>

          {/* Forecast Chart */}
          {chartData.length > 0 && (
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Daily Demand Forecast - {selectedCategory} ({forecastDays} Days)</CardTitle>
              </CardHeader>
              <CardContent>
                <LineChart
                  data={chartData}
                  xKey="day"
                  lines={[
                    { key: "forecast", color: "#3B82F6", name: "Forecasted Demand" }
                  ]}
                />
              </CardContent>
            </Card>
          )}

          {/* Confidence Interval & Recommendations */}
          <div className="grid gap-6 md:grid-cols-2 mb-6">
            {/* Confidence Interval */}
            <Card>
              <CardHeader>
                <CardTitle>Confidence Interval (95%)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-400">Lower Bound</span>
                      <span className="text-lg font-bold text-white">{formatNumber(confidenceLower)}</span>
                    </div>
                    <div className="w-full bg-dark-border rounded-full h-2">
                      <div className="bg-blue-400 h-2 rounded-full" style={{ width: "33%" }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-400">Expected Forecast</span>
                      <span className="text-lg font-bold text-primary-400">{formatNumber(totalForecast)}</span>
                    </div>
                    <div className="w-full bg-dark-border rounded-full h-2">
                      <div className="bg-primary-500 h-2 rounded-full" style={{ width: "66%" }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-400">Upper Bound</span>
                      <span className="text-lg font-bold text-white">{formatNumber(confidenceUpper)}</span>
                    </div>
                    <div className="w-full bg-dark-border rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: "100%" }}></div>
                    </div>
                  </div>

                  <p className="text-sm text-slate-400 mt-4">
                    There's a 95% probability that actual demand will fall between the lower and upper bounds.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Inventory Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle>Inventory Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-primary-500/10 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-primary-300">Safety Stock</p>
                      <p className="text-xs text-primary-400 mt-1">Buffer for demand variability</p>
                    </div>
                    <span className="text-2xl font-bold text-primary-400">{formatNumber(safetyStock)}</span>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-warning-500/10 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-warning-300">Reorder Point</p>
                      <p className="text-xs text-warning-400 mt-1">When to place new order</p>
                    </div>
                    <span className="text-2xl font-bold text-warning-400">{formatNumber(reorderPoint)}</span>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-accent-500/10 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-accent-300">Daily Average</p>
                      <p className="text-xs text-accent-400 mt-1">Expected daily demand</p>
                    </div>
                    <span className="text-2xl font-bold text-accent-400">{formatNumber(dailyAvg)}</span>
                  </div>

                  <div className={`p-3 rounded-lg ${
                    stockoutRisk === "high" ? "bg-danger-500/10" :
                    stockoutRisk === "medium" ? "bg-warning-500/10" : "bg-accent-500/10"
                  }`}>
                    <p className={`text-sm font-medium ${
                      stockoutRisk === "high" ? "text-danger-300" :
                      stockoutRisk === "medium" ? "text-warning-300" : "text-accent-300"
                    }`}>
                      {stockoutRisk === "high" ? "⚠️ High Risk - Order Immediately" :
                       stockoutRisk === "medium" ? "⚡ Medium Risk - Monitor Closely" :
                       "✅ Low Risk - Stock Levels Healthy"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Insights */}
          <Card>
            <CardHeader>
              <CardTitle>Forecasting Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-primary-500/10 rounded-lg">
                  <div className="text-primary-400 mt-1">ℹ️</div>
                  <div>
                    <p className="font-medium text-primary-300">SARIMA Model</p>
                    <p className="text-sm text-primary-400 mt-1">
                      This forecast uses SARIMA (Seasonal AutoRegressive Integrated Moving Average) time series analysis
                      trained on historical sales data for <strong>{selectedCategory}</strong>.
                    </p>
                  </div>
                </div>

                {stockoutRisk === "high" && (
                  <div className="flex items-start gap-3 p-3 bg-danger-500/10 rounded-lg">
                    <div className="text-danger-400 mt-1">⚠️</div>
                    <div>
                      <p className="font-medium text-danger-300">Action Required</p>
                      <p className="text-sm text-danger-400 mt-1">
                        High stockout risk detected. Consider expediting orders or increasing safety stock levels
                        to prevent potential stockouts in the next {forecastDays} days.
                      </p>
                    </div>
                  </div>
                )}

                {stockoutRisk === "medium" && (
                  <div className="flex items-start gap-3 p-3 bg-warning-500/10 rounded-lg">
                    <div className="text-warning-400 mt-1">⚡</div>
                    <div>
                      <p className="font-medium text-warning-300">Monitor Inventory</p>
                      <p className="text-sm text-warning-400 mt-1">
                        Medium stockout risk. Keep close watch on inventory levels and prepare to reorder
                        when stock reaches {formatNumber(reorderPoint)} units.
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex items-start gap-3 p-3 bg-accent-500/10 rounded-lg">
                  <div className="text-accent-400 mt-1">✅</div>
                  <div>
                    <p className="font-medium text-accent-300">Forecast Summary</p>
                    <p className="text-sm text-accent-400 mt-1">
                      Expected demand over next {forecastDays} days: <strong>{formatNumber(totalForecast)} units</strong>
                      {" "}(~{formatNumber(dailyAvg)} units/day).
                      Maintain minimum stock of {formatNumber(safetyStock)} units as safety buffer.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
