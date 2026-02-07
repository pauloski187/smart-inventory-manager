// API Response Types

export interface DashboardSummary {
  total_revenue_mtd: number;
  total_revenue_ytd: number;
  total_orders: number;
  avg_order_value: number;
  profit_margin: number;
  low_stock_alerts: number;
  dead_stock_value: number;
  top_products: Array<{
    product_name: string;
    count: number;
  }>;
}

export interface MonthlyTrend {
  month: string;
  revenue: number;
  profit: number;
  units_sold: number;
  order_count: number;
  profit_margin: number;
  avg_order_value: number;
  loss: number;
  revenue_growth_pct: number;
  profit_growth_pct: number;
}

export interface MonthlySalesTrend {
  monthly_trends: MonthlyTrend[];
  summary: {
    total_revenue: number;
    total_profit: number;
    avg_monthly_revenue: number;
    avg_profit_margin: number;
    months_analyzed: number;
  };
}

export interface Category {
  category: string;
  revenue: number;
  profit: number;
  units_sold: number;
  revenue_share: number;
  profit_share: number;
  profit_margin: number;
  product_count: number;
  order_count: number;
  avg_order_value: number;
}

export interface CategoryPerformance {
  categories: Category[];
  totals: {
    total_revenue: number;
    total_profit: number;
    total_categories: number;
  };
}

export interface Product {
  product_id: string;
  product_name: string;
  category: string;
  revenue: number;
  profit: number;
  units_sold: number;
  profit_margin: number;
}

export interface ProductPerformance {
  best_by_revenue: Product[];
  worst_by_revenue: Product[];
  best_by_margin: Product[];
  worst_by_margin: Product[];
  recommendations: {
    best_performers: string[];
    worst_performers: string[];
  };
}

export interface ABCProduct extends Product {
  abc_class: "A" | "B" | "C";
}

export interface ABCAnalysis {
  products: ABCProduct[];
  summary: {
    class_a_count: number;
    class_b_count: number;
    class_c_count: number;
  };
}

export interface LowStockAlert {
  product_id: string;
  product_name: string;
  category: string;
  current_stock: number;
  reorder_point: number;
  priority: "high" | "medium";
}

export interface LowStockAlerts {
  count: number;
  alerts: LowStockAlert[];
}

export interface DeadStockItem {
  product_id: string;
  product_name: string;
  category: string;
  current_stock: number;
  value_at_risk: number;
  days_since_sale: number;
  recommendation: string;
}

export interface DeadStock {
  count: number;
  total_value: number;
  items: DeadStockItem[];
}

export interface DayOfWeek {
  day_number: number;
  day: string;
  revenue: number;
  profit: number;
  order_count: number;
  avg_order_value: number;
}

export interface SalesByDayOfWeek {
  daily_distribution: DayOfWeek[];
}

export interface AvailableMonth {
  year: number;
  month: number;
  month_name: string;
  display: string;
  order_count: number;
}

export interface AvailableMonths {
  available_months: AvailableMonth[];
}

export interface MonthlyReport {
  report_period: {
    year: number;
    month: number;
    month_name: string;
    start_date: string;
    end_date: string;
  };
  summary: {
    total_revenue: number;
    total_profit: number;
    total_loss: number;
    profit_margin: number;
    total_orders: number;
    total_units_sold: number;
    avg_order_value: number;
  };
  top_products: Product[];
  worst_products: Product[];
  category_breakdown: Array<{
    category: string;
    revenue: number;
    profit: number;
    revenue_share: number;
  }>;
  daily_breakdown: Array<{
    date: string;
    revenue: number;
    profit: number;
    orders: number;
  }>;
  recommendations: Array<{
    type: "info" | "success" | "warning" | "critical";
    message: string;
  }>;
}

export interface ProductListItem {
  id: string;
  name: string;
  category: string;
  price: number;
  cost: number;
  current_stock: number;
  reorder_threshold: number;
}

export interface ProductsList {
  products: ProductListItem[];
  count: number;
}

export interface OrderListItem {
  id: string;
  order_date: string;
  product_name: string;
  category: string;
  quantity: number;
  total_revenue: number;
  total_profit: number;
  status: string;
}

export interface OrdersList {
  orders: OrderListItem[];
  count: number;
}

export interface ForecastData {
  category: string;
  forecast_90_day: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
}

export interface AllForecasts {
  generated_at: string;
  horizon_days: number;
  total_forecast_all_categories: number;
  categories: ForecastData[];
}

export interface InventoryRecommendation {
  category: string;
  forecast_90_day: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  daily_average: number;
  reorder_point: number;
  safety_stock: number;
  stockout_risk: string;
  recommended_order_quantity: number;
}
