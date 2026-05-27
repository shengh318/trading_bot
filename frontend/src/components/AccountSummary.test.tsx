import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AccountSummary from "./AccountSummary";

describe("AccountSummary", () => {
  it("renders nothing when summary is null", () => {
    const { container } = render(<AccountSummary summary={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders all financial fields", () => {
    render(
      <AccountSummary
        summary={{
          cash: 5000,
          portfolio_value: 15000,
          buying_power: 10000,
          day_pnl: 250.5,
        }}
      />,
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
      <AccountSummary
        summary={{ cash: 5000, portfolio_value: 15000, buying_power: 10000, day_pnl: -100 }}
      />,
    );
    const pnl = screen.getByText("-$100.00");
    expect(pnl.style.color).toBe("rgb(197, 34, 31)");
  });
});
