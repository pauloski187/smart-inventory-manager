"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  TrendingUp,
  Calendar,
  Trophy,
  PieChart,
  BarChart3,
  Layers,
  Bell,
  List,
  ShoppingCart,
  LineChart,
} from "lucide-react";

const navigation = [
  {
    name: "Overview",
    items: [
      { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    ],
  },
  {
    name: "Analytics",
    items: [
      { name: "Sales Trends", href: "/trends", icon: TrendingUp },
      { name: "Monthly Reports", href: "/reports", icon: Calendar },
      { name: "Product Performance", href: "/products", icon: Trophy },
      { name: "Category Performance", href: "/categories", icon: PieChart },
      { name: "Sales Patterns", href: "/patterns", icon: BarChart3 },
      { name: "Demand Forecasting", href: "/forecasting", icon: LineChart },
    ],
  },
  {
    name: "Inventory",
    items: [
      { name: "ABC Analysis", href: "/abc-analysis", icon: Layers },
      { name: "Alerts", href: "/alerts", icon: Bell },
      { name: "Products List", href: "/products-list", icon: List },
      { name: "Orders List", href: "/orders-list", icon: ShoppingCart },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-screen w-64 flex-col border-r border-dark-border bg-dark-card">
      {/* Logo */}
      <div className="flex h-16 items-center border-b border-dark-border px-6">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-primary-500/20 flex items-center justify-center">
            <LayoutDashboard className="h-4 w-4 text-primary-400" />
          </div>
          <h1 className="text-lg font-bold text-white">
            Smart<span className="text-primary-400">Inventory</span>
          </h1>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-hide">
        {navigation.map((section) => (
          <div key={section.name}>
            <h3 className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
              {section.name}
            </h3>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                      isActive
                        ? "bg-primary-500/15 text-primary-400 border-l-2 border-primary-400"
                        : "text-slate-400 hover:text-white hover:bg-dark-hover"
                    )}
                  >
                    <Icon className={cn("h-[18px] w-[18px]", isActive && "text-primary-400")} />
                    {item.name}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-dark-border p-4">
        <div className="rounded-lg bg-dark-secondary p-3">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-accent-500 animate-pulse"></div>
            <p className="text-xs font-medium text-slate-300">API Status: Online</p>
          </div>
          <p className="mt-1 text-[11px] text-slate-500">Hugging Face Spaces</p>
        </div>
      </div>
    </div>
  );
}
