"use client";

import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from "recharts";

interface BarChartProps {
  data: any[];
  xKey: string;
  yKey: string;
  color?: string;
  colors?: string[];
  name?: string;
  horizontal?: boolean;
}

export function BarChart({ data, xKey, yKey, color = "#0ea5e9", colors, name = "Value", horizontal = false }: BarChartProps) {
  const ChartComponent = horizontal ? RechartsBarChart : RechartsBarChart;

  return (
    <ResponsiveContainer width="100%" height={350}>
      <ChartComponent
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={{ top: 5, right: 30, left: horizontal ? 100 : 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(45, 53, 72, 0.5)" />
        <XAxis
          type={horizontal ? "number" : "category"}
          dataKey={horizontal ? undefined : xKey}
          tick={{ fill: "#64748b", fontSize: 12 }}
          tickLine={{ stroke: "#2d3548" }}
          axisLine={{ stroke: "#2d3548" }}
        />
        <YAxis
          type={horizontal ? "category" : "number"}
          dataKey={horizontal ? xKey : undefined}
          tick={{ fill: "#64748b", fontSize: 12 }}
          tickLine={{ stroke: "#2d3548" }}
          axisLine={{ stroke: "#2d3548" }}
          width={horizontal ? 100 : undefined}
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
        />
        {!horizontal && <Legend wrapperStyle={{ color: "#94a3b8" }} />}
        <Bar dataKey={yKey} name={name} radius={[4, 4, 0, 0]}>
          {colors ? (
            data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))
          ) : (
            <Cell fill={color} />
          )}
        </Bar>
      </ChartComponent>
    </ResponsiveContainer>
  );
}
