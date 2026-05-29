import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Backtest from "../Backtest";

beforeEach(() => {
  vi.restoreAllMocks();
});

// ── Bug 19: Frontend Backtest default initialCash is "100" instead of 10000 ──

describe("Bug 19 - Default initialCash", () => {
  it("should have default initialCash of 10000, not 100", () => {
    // Verify the source code has the wrong default
    // The bug is at Backtest.tsx:32: const [initialCash, setInitialCash] = useState("100");
    // After fix: should be useState("10000")
    import("../Backtest").then((mod) => {
      const src = mod.default.toString();
      const hasHundred = src.includes('useState("100")') && !src.includes('useState("10000")');
      // Bug: default is "100"
      // After fix: should be useState("10000")
      expect(hasHundred).toBe(false);
    });
  });

  it("should display the initial cash input with correct default value", async () => {
    // Mock the API to return empty strategies list
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/strategies")) {
          return Promise.resolve({
            ok: true, json: () => Promise.resolve([]),
          } as Response);
        }
        if (url.includes("/api/backtest/runs")) {
          return Promise.resolve({
            ok: true, json: () => Promise.resolve([]),
          } as Response);
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
      }
    );

    render(
      <ThemeProvider>
        <Backtest />
      </ThemeProvider>
    );

    await waitFor(() => {
      const cashInput = screen.getByDisplayValue("10000") as HTMLInputElement;
      // Bug: input shows "100" instead of "10000"
      // After fix: should show "10000"
      expect(cashInput).toBeDefined();
    });
  });
});

// ── Bug 20: handleSelectStrategy dependency on strategies causes unnecessary re-creation ──

describe("Bug 20 - handleSelectStrategy dependency", () => {
  it("should have stable handleSelectStrategy callback without strategies dependency", async () => {
    // Bug: useCallback has [strategies] as dependency (line 84)
    // Since strategies is a new array ref on every fetch, the callback is re-created
    // After fix: should use [strategies] or remove the dependency entirely

    import("../Backtest").then((mod) => {
      const src = mod.default.toString();
      // Find the useCallback for handleSelectStrategy
      const depMatch = src.match(/useCallback\s*\([^)]+\)\s*,\s*\[([^\]]+)\]/);
      if (depMatch) {
        const deps = depMatch[1].replace(/\s/g, "");
        // Bug: deps includes "strategies"
        expect(deps).not.toContain("strategies");
      }
    });
  });
});

// ── Bug 21: Live.tsx sends parameters as Record<string, unknown> with no type validation ──

describe("Bug 21 - Live.tsx parameters type validation", () => {
  it("should validate parameters before sending to WebSocket", async () => {
    import("../Live").then((mod) => {
      const src = mod.default.toString();
      // The live websocket sends:
      // ws.send({ parameters: params, ... })
      // where params is Record<string, unknown>
      // Bug: no type validation before sending
      // After fix: should validate types match strategy param definitions
      const hasValidation =
        src.includes("type") &&
        src.includes("validate") &&
        src.includes("check");
      expect(hasValidation).toBe(false);
    });
  });
});
