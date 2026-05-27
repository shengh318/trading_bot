import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "../theme/ThemeContext";
import AccountSummary from "./AccountSummary";

describe("AccountSummary", () => {
  it("renders nothing when summary is null", () => {
    render(<ThemeProvider><AccountSummary summary={null} /></ThemeProvider>);
    expect(screen.queryByText("Cash")).toBeNull();
  });

  it("renders all financial fields", () => {
    render(
      <ThemeProvider><AccountSummary
        summary={{
          cash: 5000,
          portfolio_value: 15000,
          buying_power: 10000,
          day_pnl: 250.5,
        }}
      /></ThemeProvider>,
    );
    expect(screen.getByText("$5000.00")).toBeDefined();
    expect(screen.getByText("$15000.00")).toBeDefined();
    expect(screen.getByText("$10000.00")).toBeDefined();
    expect(screen.getByText("Purchasing Power")).toBeDefined();
    expect(screen.getByText("Today's Profit/Loss")).toBeDefined();
    expect(screen.getByText("+$250.50")).toBeDefined();
  });

  it("shows negative P&L in red", () => {
    render(
      <ThemeProvider><AccountSummary
        summary={{ cash: 5000, portfolio_value: 15000, buying_power: 10000, day_pnl: -100 }}
      /></ThemeProvider>,
    );
    const pnl = screen.getByText("-$100.00");
    expect(pnl.style.color).toBe("rgb(197, 34, 31)"); // colors.negative in light mode
  });
});
