import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import PositionsTable from "./PositionsTable";

describe("PositionsTable", () => {
  it("shows empty state", () => {
    render(<PositionsTable positions={[]} />);
    expect(screen.getByText("No holdings")).toBeDefined();
  });

  it("renders position rows", () => {
    render(
      <PositionsTable
        positions={[
          { symbol: "AAPL", qty: 10, avg_entry_price: 150, current_price: 155, unrealized_pl: 50, market_value: 1550 },
          { symbol: "GOOG", qty: 5, avg_entry_price: 2800, current_price: 2750, unrealized_pl: -250, market_value: 13750 },
        ]}
      />,
    );
    expect(screen.getByText("AAPL")).toBeDefined();
    expect(screen.getByText("GOOG")).toBeDefined();
    expect(screen.getByText("$50.00")).toBeDefined();
    expect(screen.getByText("$-250.00")).toBeDefined();
  });

  it("colors positive P&L green and negative red", () => {
    render(
      <PositionsTable
        positions={[
          { symbol: "AAPL", qty: 10, avg_entry_price: 150, current_price: 155, unrealized_pl: 50, market_value: 1550 },
          { symbol: "GOOG", qty: 5, avg_entry_price: 2800, current_price: 2750, unrealized_pl: -250, market_value: 13750 },
        ]}
      />,
    );
    const greenPnl = screen.getByText("$50.00");
    const redPnl = screen.getByText("$-250.00");
    expect(greenPnl.style.color).toBe("rgb(11, 128, 67)");
    expect(redPnl.style.color).toBe("rgb(197, 34, 31)");
  });
});
