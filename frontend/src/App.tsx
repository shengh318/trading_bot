import { useState } from "react";
import { useTheme } from "./theme/ThemeContext";
import Dashboard from "./pages/Dashboard";
import Backtest from "./pages/Backtest";
import Strategies from "./pages/Strategies";
import Live from "./pages/Live";
import MlLab from "./pages/MlLab";
import Correlation from "./pages/Correlation";
import Pairs from "./pages/Pairs";
import Clock from "./components/Clock";

type Tab = "dashboard" | "backtest" | "strategies" | "live" | "ml" | "correlation" | "pairs";

const TABS: { key: Tab; label: string }[] = [
  { key: "dashboard", label: "Dashboard" },
  { key: "live", label: "Live" },
  { key: "backtest", label: "Backtest" },
  { key: "strategies", label: "Strategies" },
  { key: "ml", label: "ML Lab" },
  { key: "correlation", label: "Correlation" },
  { key: "pairs", label: "Pairs" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");
  const { isDark, toggleTheme, colors } = useTheme();

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>TraderBot</h1>
        <Clock />
        <button
          onClick={toggleTheme}
          style={{
            padding: "6px 14px",
            background: colors.surface,
            color: colors.text,
            border: `1px solid ${colors.border}`,
            borderRadius: 6,
            cursor: "pointer",
            fontSize: 13,
          }}
        >
          {isDark ? "☀️ Light" : "🌙 Dark"}
        </button>
      </div>
      <nav style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            style={{
              padding: "8px 20px",
              fontWeight: activeTab === t.key ? 700 : 400,
              background: activeTab === t.key ? colors.primary : colors.tabInactive,
              color: activeTab === t.key ? "#fff" : colors.text,
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
      {activeTab === "live" && <Live />}
      {activeTab === "backtest" && <Backtest />}
      {activeTab === "strategies" && <Strategies />}
      {activeTab === "ml" && <MlLab />}
      {activeTab === "correlation" && <Correlation />}
      {activeTab === "pairs" && <Pairs />}
    </div>
  );
}
