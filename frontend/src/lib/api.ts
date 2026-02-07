// API Integration Layer

import type {
  DashboardSummary,
  MonthlySalesTrend,
  CategoryPerformance,
  ProductPerformance,
  ABCAnalysis,
  LowStockAlerts,
  DeadStock,
  SalesByDayOfWeek,
  AvailableMonths,
  MonthlyReport,
  ProductsList,
  OrdersList,
  AllForecasts,
  InventoryRecommendation,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://pauloski07-sales-inventory-forecasting.hf.space";

export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function fetchAPI<T>(endpoint: string): Promise<T> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
      },
      next: {
        revalidate: 60, // Cache for 60 seconds
      },
    });

    if (!response.ok) {
      throw new APIError(
        `API request failed: ${response.statusText}`,
        response.status,
        endpoint
      );
    }

    return response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      `Network error: ${error instanceof Error ? error.message : "Unknown error"}`,
      undefined,
      endpoint
    );
  }
}

export const api = {
  // Dashboard
  getDashboardSummary: () =>
    fetchAPI<DashboardSummary>("/analytics/dashboard/summary"),

  // Trends
  getMonthlySalesTrend: (months: number = 12) =>
    fetchAPI<MonthlySalesTrend>(`/analytics/monthly-sales-trend?months=${months}`),

  // Categories
  getCategoryPerformance: () =>
    fetchAPI<CategoryPerformance>("/analytics/category-performance"),

  // Products
  getProductPerformance: (limit: number = 10) =>
    fetchAPI<ProductPerformance>(`/analytics/product-performance?limit=${limit}`),

  // ABC Analysis
  getABCAnalysis: (limit: number = 100) =>
    fetchAPI<ABCAnalysis>(`/analytics/abc-analysis?limit=${limit}`),

  // Alerts
  getLowStockAlerts: () =>
    fetchAPI<LowStockAlerts>("/analytics/inventory/low-stock"),

  getDeadStock: () =>
    fetchAPI<DeadStock>("/analytics/inventory/dead-stock"),

  // Patterns
  getSalesByDayOfWeek: () =>
    fetchAPI<SalesByDayOfWeek>("/analytics/sales-by-day-of-week"),

  // Reports
  getAvailableMonths: () =>
    fetchAPI<AvailableMonths>("/analytics/available-months"),

  getMonthlyReport: (year: number, month: number) =>
    fetchAPI<MonthlyReport>(`/analytics/monthly-report/${year}/${month}`),

  // Data
  getProducts: (limit: number = 100) =>
    fetchAPI<ProductsList>(`/products?limit=${limit}`),

  getOrders: (limit: number = 100) =>
    fetchAPI<OrdersList>(`/orders?limit=${limit}`),

  // Forecasting
  getAllForecasts: (horizon: number = 90) =>
    fetchAPI<AllForecasts>(`/forecast/forecasts/all?horizon=${horizon}`),

  getCategoryForecast: (category: string, includeDaily: boolean = true) =>
    fetchAPI<any>(`/forecast/forecast/${encodeURIComponent(category)}?include_daily=${includeDaily}`),

  getInventoryRecommendations: () =>
    fetchAPI<InventoryRecommendation[]>("/forecast/inventory-recommendations"),

  // Health
  healthCheck: () =>
    fetchAPI<{ status: string; service: string; version: string }>("/health"),
};

// Helper function to format currency
export function formatCurrency(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return "$0.00";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

// Helper function to format percentage
export function formatPercent(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return "0.00%";
  return `${value.toFixed(2)}%`;
}

// Helper function to format large numbers
export function formatNumber(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return "0";
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toFixed(0);
}
