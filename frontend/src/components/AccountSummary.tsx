import { memo } from "react";
import { useTheme } from "../theme/ThemeContext";
import type { AccountSummary as AccountSummaryType } from "../api/client";

interface Props {
  summary: AccountSummaryType | null;
}

function AccountSummary({ summary }: Props) {
  const { colors } = useTheme();
  if (!summary) return null;
  return (
    <div
      style={{
        display: "flex",
        gap: 24,
        padding: "12px 20px",
        background: colors.surface,
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
        <strong>Purchasing Power</strong>
        <div>{`$${summary.buying_power.toFixed(2)}`}</div>
      </div>
      <div>
        <strong>Today's Profit/Loss</strong>
        <div
          style={{
            color: summary.day_pnl >= 0 ? colors.positive : colors.negative,
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

export default memo(AccountSummary);

