import { Suspense } from "react";
import { api, formatCurrency, formatPercent } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { PieChart } from "@/components/charts/PieChart";
import { LoadingSkeleton } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#6366F1", "#EC4899", "#8B5CF6", "#14B8A6"];

async function CategoryContent() {
  try {
    const data = await api.getCategoryPerformance();

    const pieData = data.categories.map((cat) => ({
      name: cat.category,
      value: cat.revenue_share,
    }));

    return (
      <>
        {/* Summary Cards */}
        <div className="grid gap-6 md:grid-cols-3 mb-8">
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm font-medium text-slate-400">Total Revenue</p>
              <h3 className="mt-2 text-2xl font-bold text-white">
                {formatCurrency(data.totals.total_revenue)}
              </h3>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm font-medium text-slate-400">Total Profit</p>
              <h3 className="mt-2 text-2xl font-bold text-accent-400">
                {formatCurrency(data.totals.total_profit)}
              </h3>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm font-medium text-slate-400">Categories</p>
              <h3 className="mt-2 text-2xl font-bold text-primary-400">
                {data.totals.total_categories}
              </h3>
            </CardContent>
          </Card>
        </div>

        {/* Charts and Table */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Pie Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Revenue Share by Category</CardTitle>
            </CardHeader>
            <CardContent>
              <PieChart data={pieData} colors={COLORS} />
            </CardContent>
          </Card>

          {/* Top Categories Table */}
          <Card>
            <CardHeader>
              <CardTitle>Category Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {data.categories.slice(0, 5).map((category, index) => (
                  <div key={index} className="flex items-center justify-between pb-4 border-b border-dark-border last:border-0">
                    <div>
                      <p className="font-medium text-white">{category.category}</p>
                      <p className="text-sm text-slate-500 mt-1">
                        {category.order_count.toLocaleString()} orders
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-white">
                        {formatCurrency(category.revenue)}
                      </p>
                      <p className="text-sm text-accent-400 mt-1">
                        {formatPercent(category.profit_margin)} margin
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Full Category Table */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>All Categories</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-dark-border">
                  <tr className="text-left">
                    <th className="pb-3 font-semibold">Category</th>
                    <th className="pb-3 font-semibold text-right">Revenue</th>
                    <th className="pb-3 font-semibold text-right">Profit</th>
                    <th className="pb-3 font-semibold text-right">Units Sold</th>
                    <th className="pb-3 font-semibold text-right">Orders</th>
                    <th className="pb-3 font-semibold text-right">Margin</th>
                    <th className="pb-3 font-semibold text-right">Market Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                  {data.categories.map((category, index) => (
                    <tr key={index} className="hover:bg-dark-hover">
                      <td className="py-3 font-medium">{category.category}</td>
                      <td className="py-3 text-right">{formatCurrency(category.revenue)}</td>
                      <td className="py-3 text-right text-accent-400">{formatCurrency(category.profit)}</td>
                      <td className="py-3 text-right">{category.units_sold.toLocaleString()}</td>
                      <td className="py-3 text-right">{category.order_count.toLocaleString()}</td>
                      <td className="py-3 text-right">{formatPercent(category.profit_margin)}</td>
                      <td className="py-3 text-right font-medium text-primary-400">
                        {formatPercent(category.revenue_share)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </>
    );
  } catch (error) {
    return <ErrorMessage message={error instanceof Error ? error.message : "Failed to load data"} />;
  }
}

export default function CategoriesPage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Category Performance</h1>
        <p className="mt-2 text-slate-400">Analyze revenue and profit across product categories</p>
      </div>

      <Suspense fallback={<LoadingSkeleton />}>
        <CategoryContent />
      </Suspense>
    </div>
  );
}
