import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Smart Inventory Manager",
  description: "AI-powered inventory management and demand forecasting system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex h-screen overflow-hidden bg-dark-base">
          <Sidebar />
          <main className="flex-1 overflow-y-auto bg-dark-base">
            <div className="container mx-auto px-6 py-8 max-w-7xl animate-fade-in">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
