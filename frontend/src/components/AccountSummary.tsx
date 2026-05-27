import type { AccountSummary as AccountSummaryType } from "../api/client";

interface Props {
  summary: AccountSummaryType | null;
}

export default function AccountSummary({ summary }: Props) {
  if (!summary) return null;
  return (
    <div
      style={{
        display: "flex",
        gap: 24,
        padding: "12px 20px",
        background: "#f8f9fa",
        borderRadius: 8,
        marginBottom: 16,
        fontSize: 14,
      }}
    >
      <div>
        <strong>Cash</strong>
        <div>{`$${summary.cash.toFixed(2)}`}</div>
      </div>
      <div>
        <strong>Portfolio Value</strong>
        <div>{`$${summary.portfolio_value.toFixed(2)}`}</div>
      </div>
      <div>
        <strong>Buying Power</strong>
        <div>{`$${summary.buying_power.toFixed(2)}`}</div>
      </div>
      <div>
        <strong>Day P&L</strong>
        <div
          style={{
            color: summary.day_pnl >= 0 ? "#0b8043" : "#c5221f",
          }}
        >
          {summary.day_pnl >= 0
            ? `+$${summary.day_pnl.toFixed(2)}`
            : `-$${Math.abs(summary.day_pnl).toFixed(2)}`}
        </div>
      </div>
    </div>
  );
}
