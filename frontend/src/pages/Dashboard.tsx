import { useEffect, useMemo, useState } from "react";
import { api, type AccountSummary, type Position, type Order } from "../api/client";
import AccountSummaryWidget from "../components/AccountSummary";
import PortfolioChart from "../components/PortfolioChart";
import PositionsTable from "../components/PositionsTable";
import OrderHistory from "../components/OrderHistory";

export default function Dashboard() {
  const [summary, setSummary] = useState<AccountSummary | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [equity, setEquity] = useState<{ timestamp: number; equity: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    setError("");
    Promise.all([
      api.getAlpacaAccount(),
      api.getAlpacaPositions(),
      api.getAlpacaOrders(),
      api.getAlpacaPortfolioHistory(),
    ])
      .then(([acct, pos, ords, hist]) => {
        setSummary(acct);
        setPositions(pos);
        setOrders(ords);
        setEquity(hist);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setLoading(false));
  }, []);

  const chartData = useMemo(
    () =>
      equity.map((p) => ({
        time: p.timestamp as unknown as import("lightweight-charts").Time,
        value: p.equity,
      })),
    [equity],
  );

  if (loading) {
    return <div style={{ padding: 24, color: "#888" }}>Loading Alpaca account data…</div>;
  }

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <p style={{ color: "#e44" }}>Could not connect to Alpaca paper trading.</p>
        <p style={{ fontFamily: "monospace", fontSize: 13, color: "#888" }}>{error}</p>
        <p style={{ fontSize: 13, marginTop: 16 }}>
          Make sure your Alpaca API keys are set in <code>.env</code> and the backend is running.
        </p>
      </div>
    );
  }

  return (
    <div>
      <AccountSummaryWidget summary={summary} />
      <PortfolioChart data={chartData} />
      <div style={{ display: "flex", gap: 24 }}>
        <div style={{ flex: 1 }}>
          <PositionsTable positions={positions} />
        </div>
        <div style={{ flex: 2 }}>
          <OrderHistory orders={orders} />
        </div>
      </div>
    </div>
  );
}
