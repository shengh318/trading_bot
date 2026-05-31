import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Pairs from "../Pairs";

const emptyPairResponse = {
  status: "ok",
  pair: {
    ticker_a: "XOM", ticker_b: "CVX",
    correlations: [],
    cointegration: { method: "engle-granger", p_value: 1, test_statistic: 0, critical_values: {}, hedge_ratio: 0, is_cointegrated: false },
    spread: { mean: 0, std: 0, current_zscore: 0, half_life: 0, hurst_exponent: 0.5, spread_series: [], zscore_series: [] },
    walk_forward: { avg_half_life: 0, cointegration_percentage: 0, avg_spread_sharpe: 0, hedge_ratio_stability: 0, num_folds: 0 },
    backtest: { total_return_pct: 0, sharpe_ratio: 0, max_drawdown_pct: 0, win_rate_pct: 0, num_trades: 0, avg_holding_period: 0, turnover: 0, final_equity: 0, equity_curve: [] },
    regime: { current_regime: "mean_reverting", trading_allowed: true, signal_suppressed: false, structural_break: false, correlation_breakdown: false, spread_variance_expansion: false, vix_level: 15, regime_summary: {} },
    score: 0,
  },
};

const cointegratedPairResponse = {
  status: "ok",
  pair: {
    ticker_a: "AAPL", ticker_b: "MSFT",
    correlations: [],
    cointegration: { method: "engle-granger", p_value: 0.001, test_statistic: -5, critical_values: {}, hedge_ratio: 1.5, is_cointegrated: true },
    spread: { mean: 0, std: 1, current_zscore: 0.5, half_life: 10, hurst_exponent: 0.3, is_stationary: true, adf_pvalue: 0.001, mean_reversion_speed: 0.07, persistence: 0.4, variance_ratio: 0.8, spread_autocorr_5: 0.2, spread_series: [], zscore_series: [] },
    walk_forward: { avg_train_p_value: 0.01, avg_oos_p_value: 0.02, avg_half_life: 12, cointegration_percentage: 90, avg_spread_sharpe: 1.5, hedge_ratio_stability: 0.8, num_folds: 5 },
    backtest: { total_return_pct: 25, annualised_return_pct: 8, sharpe_ratio: 1.8, sortino_ratio: 2.1, calmar_ratio: 1.5, max_drawdown_pct: -12, win_rate_pct: 60, num_trades: 50, avg_holding_period: 15, turnover: 0.5, final_equity: 12500, profit_factor: 1.8, exposure_pct: 70, beta_to_market: 0.3, equity_curve: [], drawdown_series: [] },
    regime: { current_regime: "mean_reverting", trading_allowed: true, signal_suppressed: false, structural_break: false, correlation_breakdown: false, spread_variance_expansion: false, vix_level: 15, regime_summary: { mean_reverting: 0.8, trending: 0.1, volatile: 0.05, choppy: 0.05 }, cusum_break_detected: false, chow_break_detected: false, bai_perron_breaks: [], num_structural_breaks: 0 },
    score: 0.85,
  },
};

function mockFetch(response: unknown) {
  return () => Promise.resolve({ ok: true, json: () => Promise.resolve(response) } as Response);
}

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(globalThis, "fetch").mockImplementation(mockFetch(emptyPairResponse));
});

describe("Pairs Page", () => {
  it("renders section navigation buttons", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    expect(screen.getByText("Single Pair")).toBeDefined();
    expect(screen.getByText("Rank Pairs")).toBeDefined();
    expect(screen.getByText("Heatmap")).toBeDefined();
  });

  it("renders date and significance controls", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    expect(screen.getByText("Start Date")).toBeDefined();
    expect(screen.getByText("End Date")).toBeDefined();
    expect(screen.getByText("α")).toBeDefined();
  });

  it("defaults to Analyze section with ticker inputs", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    expect(screen.getByDisplayValue("XOM")).toBeDefined();
    expect(screen.getByDisplayValue("CVX")).toBeDefined();
    expect(screen.getByText("Analyze Pair")).toBeDefined();
  });

  it("shows help section on click", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("How to read this page"));
    expect(screen.getByText(/Finds stock pairs/)).toBeDefined();
  });

  it("switches to Rank section on click", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getAllByText("Rank Pairs")[0]);
    expect(screen.getByText(/Pairs \(one per line/)).toBeDefined();
    expect(screen.getAllByText("Rank Pairs")[1]).toBeDefined();
  });

  it("switches to Heatmap section on click", () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("Heatmap"));
    expect(screen.getByText(/Tickers \(comma-separated\)/)).toBeDefined();
    expect(screen.getByText("Build Heatmap")).toBeDefined();
  });

  it("analyze button is clickable", async () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    const btn = screen.getByText("Analyze Pair");
    expect(btn).toBeDefined();
    fireEvent.click(btn);
  });

  it("shows pair result after analyze", async () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("Analyze Pair"));
    await waitFor(() => {
      expect(screen.getByText("XOM / CVX")).toBeDefined();
    });
  });

  it("shows NOT COINTEGRATED badge when pair is not cointegrated", async () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("Analyze Pair"));
    await waitFor(() => {
      expect(screen.getByText("NOT COINTEGRATED")).toBeDefined();
    });
  });

  it("renders COINTEGRATED badge when pair is cointegrated", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(mockFetch(cointegratedPairResponse));
    render(<ThemeProvider><Pairs /></ThemeProvider>);

    const btn = screen.getByText("Analyze Pair");
    fireEvent.click(btn);
    await waitFor(() => {
      expect(screen.getByText("COINTEGRATED")).toBeDefined();
    });
  });

  it("renders RegimeBadge with regime name", async () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("Analyze Pair"));
    await waitFor(() => {
      expect(screen.getByText("MEAN REVERTING")).toBeDefined();
    });
  });

  it("shows walk-forward section with avg_oos_p_value", async () => {
    render(<ThemeProvider><Pairs /></ThemeProvider>);
    fireEvent.click(screen.getByText("Analyze Pair"));
    await waitFor(() => {
      expect(screen.getByText("Walk-Forward Validation")).toBeDefined();
    });
  });
});
