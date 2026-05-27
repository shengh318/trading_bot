import { useEffect, useState } from "react";
import { api, type AccountSummary, type Position, type Order, type EquityPoint } from "../api/client";
import AccountSummaryWidget from "../components/AccountSummary";
import PortfolioChart from "../components/PortfolioChart";
import PositionsTable from "../components/PositionsTable";
import OrderHistory from "../components/OrderHistory";

export default function Dashboard() {
  const [summary, setSummary] = useState<AccountSummary | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [equity, setEquity] = useState<EquityPoint[]>([]);

  useEffect(() => {
    api.getAccountSummary().then(setSummary);
    api.getPositions().then(setPositions);
    api.getOrders().then(setOrders);
    api.getEquityCurve().then(setEquity);
  }, []);

  const chartData = equity.map((p) => ({
    time: (new Date(p.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
    value: p.total_equity,
  }));

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
