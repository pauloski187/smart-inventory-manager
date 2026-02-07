// Product Performance Page - Template for user to expand
import { Suspense } from "react";
import { api, formatCurrency, formatPercent } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LoadingSkeleton } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";

async function ProductsContent() {
  try {
    const data = await api.getProductPerformance(10);

    return (
      <div className="space-y-6">
        {/* Best Performers */}
        <Card>
          <CardHeader>
            <CardTitle>Top 10 Products by Revenue</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-dark-border">
                  <tr className="text-left">
                    <th className="pb-3">Rank</th>
                    <th className="pb-3">Product</th>
                    <th className="pb-3">Category</th>
                    <th className="pb-3 text-right">Revenue</th>
                    <th className="pb-3 text-right">Profit</th>
                    <th className="pb-3 text-right">Margin</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                  {data.best_by_revenue.map((product, index) => (
                    <tr key={index} className="hover:bg-dark-hover">
                      <td className="py-3">
                        <Badge variant="success">#{index + 1}</Badge>
                      </td>
                      <td className="py-3 font-medium">{product.product_name}</td>
                      <td className="py-3 text-slate-400">{product.category}</td>
                      <td className="py-3 text-right font-semibold">
                        {formatCurrency(product.revenue)}
                      </td>
                      <td className="py-3 text-right text-accent-400">
                        {formatCurrency(product.profit)}
                      </td>
                      <td className="py-3 text-right">{formatPercent(product.profit_margin)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Worst Performers */}
        <Card>
          <CardHeader>
            <CardTitle>Bottom 10 Products by Revenue</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-dark-border">
                  <tr className="text-left">
                    <th className="pb-3">Product</th>
                    <th className="pb-3">Category</th>
                    <th className="pb-3 text-right">Revenue</th>
                    <th className="pb-3 text-right">Profit</th>
                    <th className="pb-3 text-right">Margin</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                  {data.worst_by_revenue.map((product, index) => (
                    <tr key={index} className="hover:bg-dark-hover">
                      <td className="py-3 font-medium">{product.product_name}</td>
                      <td className="py-3 text-slate-400">{product.category}</td>
                      <td className="py-3 text-right">{formatCurrency(product.revenue)}</td>
                      <td className="py-3 text-right">{formatCurrency(product.profit)}</td>
                      <td className="py-3 text-right">{formatPercent(product.profit_margin)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  } catch (error) {
    return <ErrorMessage message={error instanceof Error ? error.message : "Failed to load data"} />;
  }
}

export default function ProductsPage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Product Performance</h1>
        <p className="mt-2 text-slate-400">Best and worst performing products by revenue</p>
      </div>
      <Suspense fallback={<LoadingSkeleton />}>
        <ProductsContent />
      </Suspense>
    </div>
  );
}
