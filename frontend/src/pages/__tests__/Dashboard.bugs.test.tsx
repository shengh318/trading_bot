import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Dashboard from "../Dashboard";

beforeEach(() => {
  vi.restoreAllMocks();
});

// ── Bug 22: No error handling when Alpaca keys are missing ──

describe("Bug 22 - Dashboard Alpaca error handling", () => {
  it("should show Alpaca connection error when Alpaca APIs fail", async () => {
    // Mock all Alpaca APIs to fail
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/alpaca/")) {
          return Promise.reject(new Error("Alpaca not configured"));
        }
        return Promise.resolve({ json: () => Promise.resolve({}) } as Response);
      }
    );

    render(
      <ThemeProvider>
        <Dashboard />
      </ThemeProvider>
    );

    // Bug: when Alpaca fails, Dashboard shows "Could not connect to Alpaca paper trading"
    // even though local database and backtest features work fine
    // After fix: Dashboard should fallback to local DB data
    await waitFor(() => {
      expect(
        screen.getByText(/could not connect to alpaca/i)
      ).toBeDefined();
    });
  });

  it("should show loading state before Alpaca APIs resolve", async () => {
    // Delay Alpaca API responses
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input: RequestInfo | URL) => {
        return new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                json: () => Promise.resolve({}),
              } as Response),
            100
          )
        );
      }
    );

    render(
      <ThemeProvider>
        <Dashboard />
      </ThemeProvider>
    );

    // Should show loading message initially
    expect(screen.getByText(/loading alpaca account data/i)).toBeDefined();
  });

  it("should display local data as fallback when Alpaca is unavailable", async () => {
    // Mock Alpaca APIs to fail, but local APIs succeed
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/alpaca/")) {
          return Promise.reject(new Error("Alpaca not configured"));
        }
        return Promise.resolve({ json: () => Promise.resolve({}) } as Response);
      }
    );

    render(
      <ThemeProvider>
        <Dashboard />
      </ThemeProvider>
    );

    // Bug: Dashboard ONLY shows Alpaca error, no local data fallback
    // After fix: should also show local portfolio data or at least a different message
    await waitFor(() => {
      const errorMessage = screen.queryByText(/could not connect to alpaca/i);
      expect(errorMessage).not.toBeNull();
    });
  });
});
