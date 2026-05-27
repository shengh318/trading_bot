import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.toString().includes("/api/strategies")) {
      return Promise.resolve({ json: () => Promise.resolve([]) } as Response);
    }
    if (url.toString().includes("/api/portfolio/summary")) {
      return Promise.resolve({
        json: () =>
          Promise.resolve({
            cash: 10000,
            portfolio_value: 15000,
            buying_power: 20000,
            day_pnl: 100,
          }),
      } as Response);
    }
    if (url.toString().includes("/api/positions")) {
      return Promise.resolve({ json: () => Promise.resolve([]) } as Response);
    }
    if (url.toString().includes("/api/orders")) {
      return Promise.resolve({ json: () => Promise.resolve([]) } as Response);
    }
    if (url.toString().includes("/api/portfolio/equity-curve")) {
      return Promise.resolve({ json: () => Promise.resolve([]) } as Response);
    }
    if (url.toString().includes("/api/backtest/runs")) {
      return Promise.resolve({ json: () => Promise.resolve([]) } as Response);
    }
    return Promise.resolve({ json: () => Promise.resolve({}) } as Response);
  });
});

describe("App", () => {
  it("renders title and nav tabs", () => {
    render(<App />);
    expect(screen.getByText("TraderBot")).toBeDefined();
    expect(screen.getByText("Dashboard")).toBeDefined();
    expect(screen.getByText("Backtest")).toBeDefined();
    expect(screen.getByText("Strategies")).toBeDefined();
  });

  it("defaults to Dashboard tab", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("No positions")).toBeDefined();
    });
  });

  it("switches to Backtest tab on click", async () => {
    render(<App />);
    fireEvent.click(screen.getByText("Backtest"));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Run Backtest" })).toBeDefined();
    });
  });

  it("switches to Strategies tab on click", async () => {
    render(<App />);
    fireEvent.click(screen.getByText("Strategies"));
    await waitFor(() => {
      expect(screen.getByText("Available Strategies")).toBeDefined();
    });
  });
});
