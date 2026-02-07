"use client";

import { PieChart as RechartsPieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

interface PieChartProps {
  data: Array<{
    name: string;
    value: number;
  }>;
  colors?: string[];
}

const DEFAULT_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#6366f1", "#ec4899", "#8b5cf6", "#14b8a6"];

export function PieChart({ data, colors = DEFAULT_COLORS }: PieChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <RechartsPieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={(entry) => `${entry.name}: ${entry.value.toFixed(1)}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="value"
          stroke="#181d2a"
          strokeWidth={2}
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: "#181d2a",
            border: "1px solid #2d3548",
            borderRadius: "8px",
            padding: "12px",
            color: "#e2e8f0",
          }}
          formatter={(value: number) => `${value.toFixed(2)}%`}
        />
        <Legend wrapperStyle={{ color: "#94a3b8" }} />
      </RechartsPieChart>
    </ResponsiveContainer>
  );
}
