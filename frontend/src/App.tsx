import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Backtest from "./pages/Backtest";
import Strategies from "./pages/Strategies";

type Tab = "dashboard" | "backtest" | "strategies";

const TABS: { key: Tab; label: string }[] = [
  { key: "dashboard", label: "Dashboard" },
  { key: "backtest", label: "Backtest" },
  { key: "strategies", label: "Strategies" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 16 }}>
      <h1 style={{ margin: 0, marginBottom: 16 }}>TraderBot</h1>
      <nav style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            style={{
              padding: "8px 20px",
              fontWeight: activeTab === t.key ? 700 : 400,
              background: activeTab === t.key ? "#1a73e8" : "#e8eaed",
              color: activeTab === t.key ? "#fff" : "#000",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            {t.label}
          </button>
        ))}
      </nav>
      {activeTab === "dashboard" && <Dashboard />}
      {activeTab === "backtest" && <Backtest />}
      {activeTab === "strategies" && <Strategies />}
    </div>
  );
}
