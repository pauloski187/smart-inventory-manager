"use client";

import { useState, useEffect, useMemo } from "react";
import { api, formatCurrency } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { LoadingSpinner } from "@/components/ui/Loading";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Badge } from "@/components/ui/Badge";
import type { ProductsList } from "@/types/api";

export default function ProductsListPage() {
  const [data, setData] = useState<ProductsList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [stockFilter, setStockFilter] = useState<"all" | "low" | "normal">("all");

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getProducts(500);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load products");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const categories = useMemo(() => {
    if (!data) return [];
    const uniqueCategories = new Set(data.products.map(p => p.category));
    return Array.from(uniqueCategories).sort();
  }, [data]);

  const filteredProducts = useMemo(() => {
    if (!data) return [];

    return data.products.filter(product => {
      const matchesSearch =
        product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.id.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesCategory = categoryFilter === "all" || product.category === categoryFilter;

      const matchesStock =
        stockFilter === "all" ||
        (stockFilter === "low" && product.current_stock <= product.reorder_threshold) ||
        (stockFilter === "normal" && product.current_stock > product.reorder_threshold);

      return matchesSearch && matchesCategory && matchesStock;
    });
  }, [data, searchTerm, categoryFilter, stockFilter]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  const lowStockCount = data.products.filter(p => p.current_stock <= p.reorder_threshold).length;
  const totalValue = data.products.reduce((sum, p) => sum + (p.price * p.current_stock), 0);
  const avgMargin = data.products.reduce((sum, p) => sum + ((p.price - p.cost) / p.price * 100), 0) / data.products.length;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Products Catalog</h1>
        <p className="mt-2 text-slate-400">Complete list of all products in inventory</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Products</p>
            <h3 className="mt-2 text-3xl font-bold text-primary-400">{data.count}</h3>
            <p className="mt-1 text-sm text-slate-500">{categories.length} categories</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Low Stock Items</p>
            <h3 className="mt-2 text-3xl font-bold text-danger-400">{lowStockCount}</h3>
            <p className="mt-1 text-sm text-slate-500">Need reordering</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Total Inventory Value</p>
            <h3 className="mt-2 text-3xl font-bold text-accent-400">{formatCurrency(totalValue)}</h3>
            <p className="mt-1 text-sm text-slate-500">At current prices</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-400">Avg Profit Margin</p>
            <h3 className="mt-2 text-3xl font-bold text-primary-400">{avgMargin.toFixed(1)}%</h3>
            <p className="mt-1 text-sm text-slate-500">Across all products</p>
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
                placeholder="Search by name, category, or ID..."
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

            {/* Stock Filter */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Stock Status</label>
              <select
                value={stockFilter}
                onChange={(e) => setStockFilter(e.target.value as "all" | "low" | "normal")}
                className="w-full px-4 py-2 border border-dark-border rounded-lg bg-dark-secondary text-white focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
              >
                <option value="all">All Products</option>
                <option value="low">Low Stock</option>
                <option value="normal">Normal Stock</option>
              </select>
            </div>
          </div>

          {/* Filter Results Info */}
          <div className="mt-4 flex items-center justify-between text-sm text-slate-400">
            <p>
              Showing <span className="font-semibold text-white">{filteredProducts.length}</span> of{" "}
              <span className="font-semibold text-white">{data.count}</span> products
            </p>
            {(searchTerm || categoryFilter !== "all" || stockFilter !== "all") && (
              <button
                onClick={() => {
                  setSearchTerm("");
                  setCategoryFilter("all");
                  setStockFilter("all");
                }}
                className="text-primary-400 hover:text-primary/80 font-medium"
              >
                Clear Filters
              </button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Products Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Products</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="pb-3 font-semibold">ID</th>
                  <th className="pb-3 font-semibold">Product Name</th>
                  <th className="pb-3 font-semibold">Category</th>
                  <th className="pb-3 text-right font-semibold">Price</th>
                  <th className="pb-3 text-right font-semibold">Cost</th>
                  <th className="pb-3 text-right font-semibold">Margin</th>
                  <th className="pb-3 text-right font-semibold">Stock</th>
                  <th className="pb-3 text-center font-semibold">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {filteredProducts.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate-500">
                      No products found matching your filters
                    </td>
                  </tr>
                ) : (
                  filteredProducts.map((product) => {
                    const margin = ((product.price - product.cost) / product.price * 100);
                    const isLowStock = product.current_stock <= product.reorder_threshold;

                    return (
                      <tr key={product.id} className="hover:bg-dark-hover">
                        <td className="py-3 text-slate-400 font-mono text-xs">{product.id}</td>
                        <td className="py-3 font-medium">{product.name}</td>
                        <td className="py-3 text-slate-400">{product.category}</td>
                        <td className="py-3 text-right">{formatCurrency(product.price)}</td>
                        <td className="py-3 text-right text-slate-400">{formatCurrency(product.cost)}</td>
                        <td className="py-3 text-right">
                          <span className={margin >= 30 ? "text-accent-400" : margin >= 15 ? "text-warning-400" : "text-danger-400"}>
                            {margin.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 text-right">
                          <span className={isLowStock ? "text-danger-400 font-semibold" : ""}>
                            {product.current_stock}
                          </span>
                          <span className="text-slate-500 text-xs ml-1">/ {product.reorder_threshold}</span>
                        </td>
                        <td className="py-3 text-center">
                          {isLowStock ? (
                            <Badge variant="danger">Low Stock</Badge>
                          ) : (
                            <Badge variant="success">In Stock</Badge>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <button
              onClick={() => setStockFilter("low")}
              className="p-4 bg-danger-500/10 hover:bg-danger-500/20 rounded-lg text-left transition-colors"
            >
              <p className="font-semibold text-danger-300">View Low Stock Items</p>
              <p className="text-sm text-danger-400 mt-1">{lowStockCount} items need attention</p>
            </button>

            <button
              onClick={() => setCategoryFilter(categories[0] || "all")}
              className="p-4 bg-primary-500/10 hover:bg-primary-500/20 rounded-lg text-left transition-colors"
            >
              <p className="font-semibold text-primary-300">Filter by Category</p>
              <p className="text-sm text-primary-400 mt-1">{categories.length} categories available</p>
            </button>

            <button
              onClick={() => {
                setSearchTerm("");
                setCategoryFilter("all");
                setStockFilter("all");
              }}
              className="p-4 bg-dark-secondary hover:bg-dark-hover rounded-lg text-left transition-colors"
            >
              <p className="font-semibold text-white">Reset All Filters</p>
              <p className="text-sm text-slate-300 mt-1">View all {data.count} products</p>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
