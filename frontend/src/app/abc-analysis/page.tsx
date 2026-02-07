"use client";

import { useState, useEffect } from "react";
import { api, formatCurrency, formatPercent } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { PieChart } from "@/components/charts/PieChart";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";
import type { ABCAnalysis } from "@/types/api";

const ABC_COLORS = {
  A: "#10B981",
  B: "#F59E0B",
  C: "#6B7280",
};

export default function ABCAnalysisPage() {
  const [data, setData] = useState<ABCAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "A" | "B" | "C">("all");

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getABCAnalysis(100);
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

  const pieData = [
    { name: "Class A", value: (data.summary.class_a_count / data.products.length) * 100 },
    { name: "Class B", value: (data.summary.class_b_count / data.products.length) * 100 },
    { name: "Class C", value: (data.summary.class_c_count / data.products.length) * 100 },
  ];

  const filteredProducts = filter === "all"
    ? data.products
    : data.products.filter(p => p.abc_class === filter);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">ABC Analysis</h1>
        <p className="mt-2 text-slate-400">Inventory classification based on revenue contribution</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card className="border-l-4 border-l-green-500">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">Class A - High Value</p>
                <h3 className="mt-2 text-3xl font-bold text-accent-400">{data.summary.class_a_count}</h3>
                <p className="mt-1 text-sm text-slate-500">Top 20% of revenue</p>
              </div>
              <Badge variant="success">Critical</Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-yellow-500">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">Class B - Medium Value</p>
                <h3 className="mt-2 text-3xl font-bold text-warning-400">{data.summary.class_b_count}</h3>
                <p className="mt-1 text-sm text-slate-500">Next 30% of revenue</p>
              </div>
              <Badge variant="warning">Moderate</Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">Class C - Low Value</p>
                <h3 className="mt-2 text-3xl font-bold text-slate-400">{data.summary.class_c_count}</h3>
                <p className="mt-1 text-sm text-slate-500">Remaining 50% of revenue</p>
              </div>
              <Badge variant="default">Low Priority</Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart and Recommendations */}
      <div className="grid gap-6 lg:grid-cols-2 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Product Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <PieChart data={pieData} colors={[ABC_COLORS.A, ABC_COLORS.B, ABC_COLORS.C]} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Management Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="border-l-4 border-l-green-500 pl-4 py-2">
                <h4 className="font-semibold text-accent-300">Class A Products</h4>
                <ul className="mt-2 space-y-1 text-sm text-slate-300">
                  <li>• Tight inventory control</li>
                  <li>• Frequent monitoring</li>
                  <li>• Accurate demand forecasting</li>
                  <li>• High service levels</li>
                </ul>
              </div>

              <div className="border-l-4 border-l-yellow-500 pl-4 py-2">
                <h4 className="font-semibold text-warning-400">Class B Products</h4>
                <ul className="mt-2 space-y-1 text-sm text-slate-300">
                  <li>• Moderate controls</li>
                  <li>• Regular reviews</li>
                  <li>• Standard service levels</li>
                </ul>
              </div>

              <div className="border-l-4 border-l-slate-500 pl-4 py-2">
                <h4 className="font-semibold text-slate-200">Class C Products</h4>
                <ul className="mt-2 space-y-1 text-sm text-slate-300">
                  <li>• Minimal controls</li>
                  <li>• Periodic reviews</li>
                  <li>• Consider stock reduction</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Products Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Product Classification</CardTitle>
            <div className="flex gap-2">
              <button
                onClick={() => setFilter("all")}
                className={`px-3 py-1 rounded text-sm font-medium ${
                  filter === "all" ? "bg-primary-500 text-white" : "bg-dark-hover text-slate-300"
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilter("A")}
                className={`px-3 py-1 rounded text-sm font-medium ${
                  filter === "A" ? "bg-green-600 text-white" : "bg-dark-hover text-slate-300"
                }`}
              >
                Class A
              </button>
              <button
                onClick={() => setFilter("B")}
                className={`px-3 py-1 rounded text-sm font-medium ${
                  filter === "B" ? "bg-yellow-600 text-white" : "bg-dark-hover text-slate-300"
                }`}
              >
                Class B
              </button>
              <button
                onClick={() => setFilter("C")}
                className={`px-3 py-1 rounded text-sm font-medium ${
                  filter === "C" ? "bg-slate-600 text-white" : "bg-dark-hover text-slate-300"
                }`}
              >
                Class C
              </button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="pb-3 font-semibold">Class</th>
                  <th className="pb-3 font-semibold">Product</th>
                  <th className="pb-3 font-semibold">Category</th>
                  <th className="pb-3 text-right font-semibold">Revenue</th>
                  <th className="pb-3 text-right font-semibold">Profit</th>
                  <th className="pb-3 text-right font-semibold">Margin</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {filteredProducts.slice(0, 50).map((product, index) => (
                  <tr key={index} className="hover:bg-dark-hover">
                    <td className="py-3">
                      <Badge
                        variant={
                          product.abc_class === "A" ? "success" :
                          product.abc_class === "B" ? "warning" : "default"
                        }
                      >
                        {product.abc_class}
                      </Badge>
                    </td>
                    <td className="py-3 font-medium">{product.product_name}</td>
                    <td className="py-3 text-slate-400">{product.category}</td>
                    <td className="py-3 text-right">{formatCurrency(product.revenue)}</td>
                    <td className="py-3 text-right text-accent-400">{formatCurrency(product.profit)}</td>
                    <td className="py-3 text-right">{formatPercent(product.profit_margin)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredProducts.length > 50 && (
            <p className="mt-4 text-sm text-center text-slate-500">
              Showing 50 of {filteredProducts.length} products
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
