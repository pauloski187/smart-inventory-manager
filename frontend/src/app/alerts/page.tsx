// Inventory Alerts Page
import { Suspense } from "react";
import { api, formatCurrency } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LoadingSkeleton } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";
import { AlertTriangle } from "lucide-react";

async function AlertsContent() {
  try {
    const [lowStock, deadStock] = await Promise.all([
      api.getLowStockAlerts(),
      api.getDeadStock(),
    ]);

    // Categorize alerts by severity
    const criticalAlerts = lowStock.alerts.filter(a => a.priority === "high" && a.current_stock === 0);
    const urgentAlerts = lowStock.alerts.filter(a => a.priority === "high" && a.current_stock > 0);
    const warningAlerts = lowStock.alerts.filter(a => a.priority === "medium");
    const healthyCount = (lowStock.alerts.length > 0) ? 0 : 100; // Simplified

    return (
      <div className="space-y-6">
        {/* Color-Coded Alert Zones */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {/* Red Zone: Critical - Order Now */}
          <Card className="border-l-4 border-l-danger-600">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-danger-600"></div>
                  <p className="text-sm font-medium text-slate-400">Order Now</p>
                </div>
                <Badge variant="danger">{criticalAlerts.length}</Badge>
              </div>
              <h3 className="mt-2 text-2xl font-bold text-danger-600">{criticalAlerts.length}</h3>
              <p className="mt-1 text-sm text-slate-500">Out of stock - Critical</p>
            </CardContent>
          </Card>

          {/* Orange Zone: Urgent - Order This Week */}
          <Card className="border-l-4 border-l-warning-600">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-warning-600"></div>
                  <p className="text-sm font-medium text-slate-400">Order This Week</p>
                </div>
                <Badge variant="warning">{urgentAlerts.length}</Badge>
              </div>
              <h3 className="mt-2 text-2xl font-bold text-warning-600">{urgentAlerts.length}</h3>
              <p className="mt-1 text-sm text-slate-500">Very low stock</p>
            </CardContent>
          </Card>

          {/* Yellow Zone: Watch Closely */}
          <Card className="border-l-4 border-l-yellow-500">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-warning-500"></div>
                  <p className="text-sm font-medium text-slate-400">Watch Closely</p>
                </div>
                <Badge variant="warning" className="bg-yellow-100 text-warning-400">{warningAlerts.length}</Badge>
              </div>
              <h3 className="mt-2 text-2xl font-bold text-warning-400">{warningAlerts.length}</h3>
              <p className="mt-1 text-sm text-slate-500">Below reorder point</p>
            </CardContent>
          </Card>

          {/* Green Zone: You're Good */}
          <Card className="border-l-4 border-l-success-600">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-success-600"></div>
                  <p className="text-sm font-medium text-slate-400">You're Good</p>
                </div>
                <Badge variant="success">✓</Badge>
              </div>
              <h3 className="mt-2 text-2xl font-bold text-success-600">Healthy</h3>
              <p className="mt-1 text-sm text-slate-500">Stock levels optimal</p>
            </CardContent>
          </Card>
        </div>

        {/* Critical Alerts - Out of Stock */}
        {criticalAlerts.length > 0 && (
          <Card className="bg-danger-500/10 border-danger-500/30">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-6 w-6 text-danger-600" />
                  <div>
                    <CardTitle className="text-danger-300">🔴 Critical: Order Immediately</CardTitle>
                    <p className="text-sm text-danger-400 mt-1">Out of stock items requiring immediate attention</p>
                  </div>
                </div>
                <Badge variant="danger">{criticalAlerts.length}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-danger-500/30">
                    <tr className="text-left">
                      <th className="pb-3 font-semibold text-danger-300">Product</th>
                      <th className="pb-3 font-semibold text-danger-300">Category</th>
                      <th className="pb-3 text-right font-semibold text-danger-300">Stock</th>
                      <th className="pb-3 text-right font-semibold text-danger-300">Reorder Point</th>
                      <th className="pb-3 font-semibold text-danger-300">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-danger-500/20">
                    {criticalAlerts.map((alert, index) => (
                      <tr key={index} className="hover:bg-danger-500/20">
                        <td className="py-3 font-medium text-danger-300">{alert.product_name}</td>
                        <td className="py-3 text-danger-400">{alert.category}</td>
                        <td className="py-3 text-right font-bold text-danger-600">0</td>
                        <td className="py-3 text-right text-danger-400">{alert.reorder_point}</td>
                        <td className="py-3">
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-danger-400">
                            ⚡ URGENT
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Urgent Alerts - Very Low Stock */}
        {urgentAlerts.length > 0 && (
          <Card className="bg-warning-500/10 border-warning-500/30">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-6 w-6 text-warning-600" />
                  <div>
                    <CardTitle className="text-warning-300">🟠 Urgent: Order This Week</CardTitle>
                    <p className="text-sm text-warning-400 mt-1">Very low stock - order within 7 days</p>
                  </div>
                </div>
                <Badge variant="warning">{urgentAlerts.length}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-warning-500/30">
                    <tr className="text-left">
                      <th className="pb-3 font-semibold text-warning-300">Product</th>
                      <th className="pb-3 font-semibold text-warning-300">Category</th>
                      <th className="pb-3 text-right font-semibold text-warning-300">Current Stock</th>
                      <th className="pb-3 text-right font-semibold text-warning-300">Reorder Point</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-warning-500/20">
                    {urgentAlerts.map((alert, index) => (
                      <tr key={index} className="hover:bg-warning-500/20">
                        <td className="py-3 font-medium text-warning-300">{alert.product_name}</td>
                        <td className="py-3 text-warning-400">{alert.category}</td>
                        <td className="py-3 text-right font-semibold text-warning-600">{alert.current_stock}</td>
                        <td className="py-3 text-right text-warning-400">{alert.reorder_point}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Warning Alerts - Watch Closely */}
        {warningAlerts.length > 0 && (
          <Card className="bg-warning-500/10 border-warning-500/30">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-5 w-5 text-warning-400" />
                  <div>
                    <CardTitle className="text-warning-300">🟡 Watch Closely</CardTitle>
                    <p className="text-sm text-warning-400 mt-1">Below reorder point - monitor inventory</p>
                  </div>
                </div>
                <Badge variant="warning" className="bg-yellow-100 text-warning-400">{warningAlerts.length}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-warning-500/30">
                    <tr className="text-left">
                      <th className="pb-3 font-semibold text-warning-300">Product</th>
                      <th className="pb-3 font-semibold text-warning-300">Category</th>
                      <th className="pb-3 text-right font-semibold text-warning-300">Current Stock</th>
                      <th className="pb-3 text-right font-semibold text-warning-300">Reorder Point</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-warning-500/20">
                    {warningAlerts.map((alert, index) => (
                      <tr key={index} className="hover:bg-warning-500/20">
                        <td className="py-3 font-medium text-warning-300">{alert.product_name}</td>
                        <td className="py-3 text-warning-400">{alert.category}</td>
                        <td className="py-3 text-right font-semibold text-warning-400">{alert.current_stock}</td>
                        <td className="py-3 text-right text-warning-400">{alert.reorder_point}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Original Low Stock Alerts - Keep as fallback */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Low Stock Alerts</CardTitle>
              <Badge variant="warning">{lowStock.count} items</Badge>
            </div>
          </CardHeader>
          <CardContent>
            {lowStock.count > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-dark-border">
                    <tr className="text-left">
                      <th className="pb-3">Priority</th>
                      <th className="pb-3">Product</th>
                      <th className="pb-3">Category</th>
                      <th className="pb-3 text-right">Current Stock</th>
                      <th className="pb-3 text-right">Reorder Point</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-border">
                    {lowStock.alerts.map((alert, index) => (
                      <tr key={index} className="hover:bg-dark-hover">
                        <td className="py-3">
                          <Badge variant={alert.priority === "high" ? "danger" : "warning"}>
                            {alert.priority}
                          </Badge>
                        </td>
                        <td className="py-3 font-medium">{alert.product_name}</td>
                        <td className="py-3 text-slate-400">{alert.category}</td>
                        <td className="py-3 text-right font-semibold text-danger-400">
                          {alert.current_stock}
                        </td>
                        <td className="py-3 text-right">{alert.reorder_point}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-4xl mb-3">✓</div>
                <p className="text-slate-400">All stock levels are healthy!</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dead Stock */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Dead Stock (90+ days no sales)</CardTitle>
              <div className="text-right">
                <Badge variant="danger">{deadStock.count} items</Badge>
                <p className="text-sm text-slate-400 mt-1">
                  Value at risk: {formatCurrency(deadStock.total_value)}
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {deadStock.count > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-dark-border">
                    <tr className="text-left">
                      <th className="pb-3">Product</th>
                      <th className="pb-3">Category</th>
                      <th className="pb-3 text-right">Stock</th>
                      <th className="pb-3 text-right">Value at Risk</th>
                      <th className="pb-3 text-right">Days Since Sale</th>
                      <th className="pb-3">Recommendation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-border">
                    {deadStock.items.map((item, index) => (
                      <tr key={index} className="hover:bg-dark-hover">
                        <td className="py-3 font-medium">{item.product_name}</td>
                        <td className="py-3 text-slate-400">{item.category}</td>
                        <td className="py-3 text-right">{item.current_stock}</td>
                        <td className="py-3 text-right font-semibold text-danger-400">
                          {formatCurrency(item.value_at_risk)}
                        </td>
                        <td className="py-3 text-right">{item.days_since_sale} days</td>
                        <td className="py-3 text-sm text-slate-400">{item.recommendation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-4xl mb-3">✓</div>
                <p className="text-slate-400">No dead stock detected!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  } catch (error) {
    return <ErrorMessage message={error instanceof Error ? error.message : "Failed to load data"} />;
  }
}

export default function AlertsPage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Inventory Alerts</h1>
        <p className="mt-2 text-slate-400">Monitor low stock and dead stock items</p>
      </div>
      <Suspense fallback={<LoadingSkeleton />}>
        <AlertsContent />
      </Suspense>
    </div>
  );
}
