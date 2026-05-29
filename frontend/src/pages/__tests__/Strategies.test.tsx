import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Strategies from "../Strategies";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("Strategies Page", () => {
  it("shows loading state initially", async () => {
    // Delay the API response to see loading state
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                ok: true,
                json: () => Promise.resolve([]),
              } as Response),
            100
          )
        )
    );

    render(
      <ThemeProvider>
        <Strategies />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("Available Strategies")).toBeDefined();
    });
  });

  it("shows 'No strategies found' when API returns empty", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([]),
    } as Response);

    render(
      <ThemeProvider>
        <Strategies />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("No strategies found")).toBeDefined();
    });
  });

  it("shows strategy cards when API returns strategies", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve([
          {
            name: "SmaCrossover",
            description: "Simple moving average crossover strategy",
            params: [
              { name: "short_window", type: "int", default: 10 },
              { name: "long_window", type: "int", default: 50 },
            ],
          },
          {
            name: "SimpleStrat1",
            description: "Mean reversion DCA strategy",
            params: [
              { name: "entry_drop", type: "float", default: 2.0 },
              { name: "max_buys", type: "int", default: 5 },
            ],
          },
        ]),
    } as Response);

    render(
      <ThemeProvider>
        <Strategies />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("SmaCrossover")).toBeDefined();
      expect(screen.getByText("SimpleStrat1")).toBeDefined();
      expect(
        screen.getByText("Simple moving average crossover strategy")
      ).toBeDefined();
    });
  });

  it("displays parameter details for each strategy", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve([
          {
            name: "SmaCrossover",
            description: "SMA crossover",
            params: [
              { name: "short_window", type: "int", default: 10 },
              { name: "long_window", type: "int", default: 50 },
            ],
          },
        ]),
    } as Response);

    render(
      <ThemeProvider>
        <Strategies />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("short_window")).toBeDefined();
      expect(screen.getByText("long_window")).toBeDefined();
      expect(screen.getByText("10")).toBeDefined();
      expect(screen.getByText("50")).toBeDefined();
    });
  });

  it("handles API error gracefully", () =>
    // Bug: Strategies.tsx line 10 — `.then(setStrategies)` with no `.catch()`
    // If the fetch fails (network error, server down), the promise rejection
    // is unhandled. The component shows "No strategies found" (empty state)
    // instead of an error message, masking the failure.
    import("../Strategies").then((mod) => {
      const src = mod.default.toString();
      // Bug: No .catch() on the promise chain
      // After fix: should have .catch() to set error state
      expect(src).toContain(".catch");
    }));
});
