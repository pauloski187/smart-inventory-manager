import { Suspense } from "react";
import { DollarSign, Package, TrendingUp, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { StatCard } from "@/components/dashboard/StatCard";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { LoadingSkeleton } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import Link from "next/link";

async function DashboardContent() {
  try {
    const data = await api.getDashboardSummary();

    return (
      <>
        {/* KPI Cards */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 mb-8">
          <StatCard
            title="Total Orders"
            value={data.total_orders}
            icon={Package}
            format="number"
          />
          <StatCard
            title="Avg Order Value"
            value={data.avg_order_value}
            icon={DollarSign}
            format="currency"
          />
          <StatCard
            title="Profit Margin"
            value={data.profit_margin}
            icon={TrendingUp}
            format="percent"
          />
          <StatCard
            title="Low Stock Alerts"
            value={data.low_stock_alerts}
            icon={AlertTriangle}
          />
        </div>

        {/* Top Products */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Top 5 Products</CardTitle>
            </CardHeader>
            <CardContent>
              {data.top_products.length > 0 ? (
                <div className="space-y-4">
                  {data.top_products.map((product, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-500/10 text-sm font-medium text-primary-400">
                          {index + 1}
                        </span>
                        <span className="font-medium text-white">{product.product_name}</span>
                      </div>
                      <Badge variant="info">{product.count} orders</Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-center text-slate-500 py-8">No data available</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Link
                  href="/trends"
                  className="flex items-center justify-between rounded-lg border border-dark-border p-4 hover:bg-dark-hover transition-colors"
                >
                  <span className="font-medium">View Sales Trends</span>
                  <TrendingUp className="h-5 w-5 text-slate-500" />
                </Link>
                <Link
                  href="/alerts"
                  className="flex items-center justify-between rounded-lg border border-dark-border p-4 hover:bg-dark-hover transition-colors"
                >
                  <span className="font-medium">Inventory Alerts</span>
                  <AlertTriangle className="h-5 w-5 text-slate-500" />
                </Link>
                <Link
                  href="/products"
                  className="flex items-center justify-between rounded-lg border border-dark-border p-4 hover:bg-dark-hover transition-colors"
                >
                  <span className="font-medium">Product Performance</span>
                  <Package className="h-5 w-5 text-slate-500" />
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </>
    );
  } catch (error) {
    return (
      <ErrorMessage
        message={error instanceof Error ? error.message : "Failed to load dashboard data"}
      />
    );
  }
}

export default function DashboardPage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="mt-2 text-slate-400">Overview of your inventory and sales performance</p>
      </div>

      <Suspense fallback={<LoadingSkeleton />}>
        <DashboardContent />
      </Suspense>
    </div>
  );
}
