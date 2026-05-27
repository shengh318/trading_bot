import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "../theme/ThemeContext";
import OrderHistory from "./OrderHistory";

describe("OrderHistory", () => {
  it("shows empty state", () => {
    render(<ThemeProvider><OrderHistory orders={[]} /></ThemeProvider>);
    expect(screen.getByText("No orders")).toBeDefined();
  });

  it("renders order rows", () => {
    render(
      <ThemeProvider><OrderHistory
        orders={[
          {
            id: "ord_abc123",
            symbol: "AAPL",
            side: "buy",
            qty: 10,
            filled_qty: 10,
            filled_avg_price: 150.5,
            status: "filled",
            type: "market",
            created_at: "2024-01-15T10:00:00Z",
            updated_at: "2024-01-15T10:00:01Z",
          },
          {
            id: "ord_def456",
            symbol: "GOOG",
            side: "sell",
            qty: 5,
            filled_qty: 5,
            filled_avg_price: null,
            status: "pending",
            type: "limit",
            created_at: "2024-01-16T10:00:00Z",
            updated_at: "2024-01-16T10:00:00Z",
          },
        ]}
      /></ThemeProvider>,
    );
    expect(screen.getByText("ord_abc1")).toBeDefined();
    expect(screen.getByText("AAPL")).toBeDefined();
    expect(screen.getByText("buy")).toBeDefined();
    expect(screen.getByText("sell")).toBeDefined();
    expect(screen.getByText("$150.50")).toBeDefined();
    expect(screen.getByText("filled")).toBeDefined();
    expect(screen.getByText("pending")).toBeDefined();
  });

  it("colors buy green and sell red", () => {
    render(
      <ThemeProvider><OrderHistory
        orders={[
          {
            id: "1", symbol: "AAPL", side: "buy", qty: 10, filled_qty: 10, filled_avg_price: 150,
            status: "filled", type: "market", created_at: "2024-01-15T10:00:00Z", updated_at: "2024-01-15T10:00:01Z",
          },
          {
            id: "2", symbol: "GOOG", side: "sell", qty: 5, filled_qty: 5, filled_avg_price: null,
            status: "pending", type: "limit", created_at: "2024-01-16T10:00:00Z", updated_at: "2024-01-16T10:00:00Z",
          },
        ]}
      /></ThemeProvider>,
    );
    expect(screen.getByText("buy").style.color).toBe("rgb(11, 128, 67)");
    expect(screen.getByText("sell").style.color).toBe("rgb(197, 34, 31)");
  });
});
