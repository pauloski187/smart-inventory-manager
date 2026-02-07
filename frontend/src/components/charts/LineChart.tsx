"use client";

import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface LineChartProps {
  data: any[];
  xKey: string;
  lines: Array<{
    key: string;
    color: string;
    name: string;
  }>;
}

export function LineChart({ data, xKey, lines }: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <RechartsLineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(45, 53, 72, 0.5)" />
        <XAxis
          dataKey={xKey}
          tick={{ fill: "#64748b", fontSize: 12 }}
          tickLine={{ stroke: "#2d3548" }}
          axisLine={{ stroke: "#2d3548" }}
        />
        <YAxis
          tick={{ fill: "#64748b", fontSize: 12 }}
          tickLine={{ stroke: "#2d3548" }}
          axisLine={{ stroke: "#2d3548" }}
          tickFormatter={(value) => {
            if (value >= 1000) return `$${(value / 1000).toFixed(0)}K`;
            return value.toLocaleString();
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#181d2a",
            border: "1px solid #2d3548",
            borderRadius: "8px",
            padding: "12px",
            color: "#e2e8f0",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(value: number) => [`${value.toLocaleString()}`, ""]}
        />
        <Legend wrapperStyle={{ color: "#94a3b8" }} />
        {lines.map((line) => (
          <Line
            key={line.key}
            type="monotone"
            dataKey={line.key}
            stroke={line.color}
            strokeWidth={2}
            name={line.name}
            dot={{ r: 3, fill: line.color }}
            activeDot={{ r: 5, fill: line.color, stroke: "#181d2a", strokeWidth: 2 }}
          />
        ))}
      </RechartsLineChart>
    </ResponsiveContainer>
  );
}
