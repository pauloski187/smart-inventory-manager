"use client";

import { useState, useEffect, useMemo } from "react";
import { api, formatCurrency } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";
import type { OrdersList } from "@/types/api";

export default function OrdersListPage() {
  const [data, setData] = useState<OrdersList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [dateRange, setDateRange] = useState<"all" | "7d" | "30d" | "90d">("all");

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getOrders(500);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load orders");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const categories = useMemo(() => {
    if (!data) return [];
    const uniqueCategories = new Set(data.orders.map(o => o.category));
    return Array.from(uniqueCategories).sort();
  }, [data]);

  const filteredOrders = useMemo(() => {
    if (!data) return [];

    const now = new Date();
    const dayMs = 24 * 60 * 60 * 1000;

    return data.orders.filter(order => {
      const matchesSearch =
        order.product_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        order.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
        order.id.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesCategory = categoryFilter === "all" || order.category === categoryFilter;

      let matchesDate = true;
      if (dateRange !== "all") {
        const orderDate = new Date(order.order_date);
        const daysAgo = {
          "7d": 7,
          "30d": 30,
          "90d": 90,
        }[dateRange];
        matchesDate = (now.getTime() - orderDate.getTime()) <= (daysAgo * dayMs);
      }

      return matchesSearch && matchesCategory && matchesDate;
    });
  }, [data, searchTerm, categoryFilter, dateRange]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  const totalRevenue = data.orders.reduce((sum, o) => sum + o.total_revenue, 0);
  const totalProfit = data.orders.reduce((sum, o) => sum + o.total_profit, 0);
  const avgOrderValue = totalRevenue / data.orders.length;
  const profitMargin = (totalProfit / totalRevenue) * 100;

  const filteredRevenue = filteredOrders.reduce((sum, o) => sum + o.total_revenue, 0);
  const filteredProfit = filteredOrders.reduce((sum, o) => sum + o.total_profit, 0);

  const exportToCSV = () => {
    if (filteredOrders.length === 0) {
      alert("No orders to export");
      return;
    }

    // Create CSV header
    const headers = ["Order ID", "Date", "Product", "Category", "Quantity", "Revenue", "Profit", "Status"];
    const csvRows = [headers.join(",")];

    // Add data rows
    filteredOrders.forEach(order => {
      const row = [
        order.id,
        new Date(order.order_date).toLocaleDateString(),
        `"${order.product_name.replace(/"/g, '""')}"`, // Escape quotes
        `"${order.category}"`,
        order.quantity,
        order.total_revenue.toFixed(2),
        order.total_profit.toFixed(2),
        order.status
      ];
      csvRows.push(row.join(","));
    });

    // Create blob and download
    const csvContent = csvRows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);

    link.setAttribute("href", url);
    link.setAttribute("download", `orders_export_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Orders History</h1>
        <p className="mt-2 text-slate-400">Complete list of all sales orders</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Orders</p>
            <h3 className="mt-2 text-3xl font-bold text-primary-400">{data.count.toLocaleString()}</h3>
            <p className="mt-1 text-sm text-slate-500">All time</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Revenue</p>
            <h3 className="mt-2 text-3xl font-bold text-accent-400">{formatCurrency(totalRevenue)}</h3>
            <p className="mt-1 text-sm text-slate-500">{formatCurrency(avgOrderValue)} avg</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Profit</p>
            <h3 className="mt-2 text-3xl font-bold text-primary-400">{formatCurrency(totalProfit)}</h3>
            <p className="mt-1 text-sm text-slate-500">{profitMargin.toFixed(1)}% margin</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Categories</p>
            <h3 className="mt-2 text-3xl font-bold text-purple-400">{categories.length}</h3>
            <p className="mt-1 text-sm text-slate-500">Product categories</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="grid gap-4 md:grid-cols-4">
            {/* Search */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">Search</label>
              <input
                type="text"
                placeholder="Search by product, category, or order ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 border border-dark-border rounded-lg bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
              />
            </div>

            {/* Category Filter */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Category</label>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="w-full px-4 py-2 border border-dark-border rounded-lg bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
              >
                <option value="all">All Categories</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>

            {/* Date Range Filter */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Date Range</label>
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value as any)}
                className="w-full px-4 py-2 border border-dark-border rounded-lg bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
              >
                <option value="all">All Time</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
              </select>
            </div>
          </div>

          {/* Filter Results Info */}
          <div className="mt-4 flex items-center justify-between text-sm">
            <div>
              <p className="text-slate-400">
                Showing <span className="font-semibold text-white">{filteredOrders.length}</span> of{" "}
                <span className="font-semibold text-white">{data.count}</span> orders
              </p>
              {filteredOrders.length > 0 && (
                <p className="text-slate-400 mt-1">
                  Revenue: <span className="font-semibold text-accent-400">{formatCurrency(filteredRevenue)}</span>
                  {" • "}
                  Profit: <span className="font-semibold text-primary-400">{formatCurrency(filteredProfit)}</span>
                </p>
              )}
            </div>
            {(searchTerm || categoryFilter !== "all" || dateRange !== "all") && (
              <button
                onClick={() => {
                  setSearchTerm("");
                  setCategoryFilter("all");
                  setDateRange("all");
                }}
                className="text-primary-400 hover:text-primary/80 font-medium"
              >
                Clear Filters
              </button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Orders Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Orders</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="pb-3 font-semibold">Order ID</th>
                  <th className="pb-3 font-semibold">Date</th>
                  <th className="pb-3 font-semibold">Product</th>
                  <th className="pb-3 font-semibold">Category</th>
                  <th className="pb-3 text-right font-semibold">Quantity</th>
                  <th className="pb-3 text-right font-semibold">Revenue</th>
                  <th className="pb-3 text-right font-semibold">Profit</th>
                  <th className="pb-3 text-center font-semibold">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {filteredOrders.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate-500">
                      No orders found matching your filters
                    </td>
                  </tr>
                ) : (
                  filteredOrders.slice(0, 100).map((order) => {
                    const margin = (order.total_profit / order.total_revenue) * 100;

                    return (
                      <tr key={order.id} className="hover:bg-dark-hover">
                        <td className="py-3 text-slate-400 font-mono text-xs">{order.id}</td>
                        <td className="py-3 text-slate-400">
                          {new Date(order.order_date).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })}
                        </td>
                        <td className="py-3 font-medium">{order.product_name}</td>
                        <td className="py-3 text-slate-400">{order.category}</td>
                        <td className="py-3 text-right">{order.quantity}</td>
                        <td className="py-3 text-right">{formatCurrency(order.total_revenue)}</td>
                        <td className="py-3 text-right">
                          <span className={margin >= 30 ? "text-accent-400" : margin >= 15 ? "text-warning-400" : "text-danger-400"}>
                            {formatCurrency(order.total_profit)}
                          </span>
                        </td>
                        <td className="py-3 text-center">
                          <Badge
                            variant={
                              order.status === "completed" ? "success" :
                              order.status === "pending" ? "warning" :
                              order.status === "cancelled" ? "danger" : "default"
                            }
                          >
                            {order.status}
                          </Badge>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
          {filteredOrders.length > 100 && (
            <p className="mt-4 text-sm text-center text-slate-500">
              Showing first 100 of {filteredOrders.length} orders
            </p>
          )}
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid gap-6 lg:grid-cols-3 mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <button
                onClick={() => setDateRange("7d")}
                className="w-full p-3 bg-primary-500/10 hover:bg-primary-500/20 rounded-lg text-left transition-colors"
              >
                <p className="font-semibold text-primary-300">Last 7 Days</p>
                <p className="text-sm text-primary-400 mt-1">View recent orders</p>
              </button>
              <button
                onClick={() => setDateRange("30d")}
                className="w-full p-3 bg-accent-500/10 hover:bg-accent-500/20 rounded-lg text-left transition-colors"
              >
                <p className="font-semibold text-accent-300">Last 30 Days</p>
                <p className="text-sm text-accent-400 mt-1">View monthly orders</p>
              </button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top Categories</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {categories.slice(0, 5).map((cat) => {
                const catOrders = data.orders.filter(o => o.category === cat);
                const catRevenue = catOrders.reduce((sum, o) => sum + o.total_revenue, 0);
                return (
                  <button
                    key={cat}
                    onClick={() => setCategoryFilter(cat)}
                    className="w-full flex items-center justify-between p-2 hover:bg-dark-hover rounded transition-colors"
                  >
                    <span className="text-sm font-medium text-white">{cat}</span>
                    <span className="text-xs text-slate-400">{catOrders.length} orders</span>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <button
                onClick={() => {
                  setSearchTerm("");
                  setCategoryFilter("all");
                  setDateRange("all");
                }}
                className="w-full p-3 bg-dark-secondary hover:bg-dark-hover rounded-lg text-left transition-colors"
              >
                <p className="font-semibold text-white">Reset Filters</p>
                <p className="text-sm text-slate-300 mt-1">View all orders</p>
              </button>
              <button
                onClick={exportToCSV}
                className="w-full p-3 bg-success-50 hover:bg-success-100 rounded-lg text-left transition-colors"
              >
                <p className="font-semibold text-accent-300">Export to CSV</p>
                <p className="text-sm text-accent-400 mt-1">
                  {filteredOrders.length} {filteredOrders.length === 1 ? "order" : "orders"}
                </p>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
