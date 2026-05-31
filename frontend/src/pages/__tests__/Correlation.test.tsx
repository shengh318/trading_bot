import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Correlation from "../Correlation";

function mockCorrelationData() {
  vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/correlation/data")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          symbol_a: "NVDA",
          symbol_b: "SPY",
          correlations: {
            "20": [{ time: "2024-01-01", value: 0.5 }, { time: "2024-01-02", value: 0.6 }],
            "60": [{ time: "2024-01-01", value: 0.4 }, { time: "2024-01-02", value: 0.5 }],
            "120": [{ time: "2024-01-01", value: 0.3 }, { time: "2024-01-02", value: 0.4 }],
          },
          cumulative_returns: {
            NVDA: [{ time: "2024-01-01", value: 100 }, { time: "2024-01-02", value: 110 }],
            SPY: [{ time: "2024-01-01", value: 100 }, { time: "2024-01-02", value: 105 }],
          },
          statistics: {
            pearson_r: 0.85, pearson_p: 0.001,
            spearman_r: 0.82, spearman_p: 0.001,
            kendall_tau: 0.65, kendall_p: 0.001,
            rolling_corr_std: 0.12, oos_corr_drop: 0.05,
          },
        }),
      } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  mockCorrelationData();
});

describe("Correlation Page", () => {
  it("renders symbol inputs with default values", () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    expect(screen.getByDisplayValue("NVDA")).toBeDefined();
    expect(screen.getByDisplayValue("SPY")).toBeDefined();
  });

  it("renders years input with default 5", () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    expect(screen.getByDisplayValue("5")).toBeDefined();
  });

  it("renders window checkboxes", () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    expect(screen.getByText("Windows:")).toBeDefined();
    expect(screen.getByText("20")).toBeDefined();
    expect(screen.getByText("60")).toBeDefined();
    expect(screen.getByText("120")).toBeDefined();
  });

  it("renders Load Data button", async () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Load Data")).toBeDefined();
    });
  });

  it("loads data on mount and displays statistics", async () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Pearson r")).toBeDefined();
      expect(screen.getByText("0.85")).toBeDefined();
      expect(screen.getByText("Spearman ρ")).toBeDefined();
      expect(screen.getByText("0.82")).toBeDefined();
      expect(screen.getByText("Kendall τ")).toBeDefined();
      expect(screen.getByText("0.65")).toBeDefined();
      expect(screen.getByText("Rolling Corr Std")).toBeDefined();
      expect(screen.getByText("0.12")).toBeDefined();
      expect(screen.getByText("OOS Corr Drop")).toBeDefined();
      expect(screen.getByText("0.05")).toBeDefined();
    });
  });

  it("shows legend with window labels", async () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText(/20-day correlation/)).toBeDefined();
      expect(screen.getAllByText(/NVDA vs SPY/).length).toBeGreaterThanOrEqual(2);
    });
  });

  it("shows loading state when fetching", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise((resolve) => setTimeout(
        () => resolve({
          ok: true,
          json: () => Promise.resolve({
            symbol_a: "NVDA", symbol_b: "SPY",
            correlations: {}, cumulative_returns: {}, statistics: {},
          }),
        } as Response),
        100
      ))
    );
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    expect(screen.getByText("Loading…")).toBeDefined();
  });

  it("shows error message on API failure", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Network error"));
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Network error")).toBeDefined();
    });
  });

  it("allows changing symbol A", () => {
    render(<ThemeProvider><Correlation /></ThemeProvider>);
    const input = screen.getByDisplayValue("NVDA") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "AMD" } });
    expect(input.value).toBe("AMD");
  });
});
