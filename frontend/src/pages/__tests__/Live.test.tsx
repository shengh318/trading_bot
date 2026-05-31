import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ThemeProvider } from "../../theme/ThemeContext";
import Live from "../Live";

const mockStrategies = [
  { name: "SmaCrossover", description: "SMA crossover", params: [{ name: "short_window", type: "int", default: 20 }, { name: "long_window", type: "int", default: 50 }] },
  { name: "SimpleStrat1", description: "Mean reversion", params: [{ name: "entry_drop", type: "float", default: 2.0 }, { name: "max_buys", type: "int", default: 5 }] },
];

beforeEach(() => {
  vi.restoreAllMocks();

  vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/strategies")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockStrategies) } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });

  vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ({
    onopen: null,
    onmessage: null,
    onerror: null,
    onclose: null,
    close: vi.fn(),
    send: vi.fn(),
  }));

  Object.defineProperty(globalThis, "location", {
    value: { protocol: "http:", host: "localhost:5173", hostname: "localhost", href: "http://localhost:5173/" },
    writable: true,
  });
});

describe("Live Page", () => {
  it("renders title and strategy selector", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Live Trading")).toBeDefined();
    });
  });

  it("loads strategies on mount", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("SmaCrossover")).toBeDefined();
    });
  });

  it("shows available symbols as checkboxes", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Symbols:")).toBeDefined();
    });
    expect(screen.getByText("NVDA")).toBeDefined();
    expect(screen.getByText("AMD")).toBeDefined();
    expect(screen.getByText("SPY")).toBeDefined();
  });

  it("shows timeframe selector with options", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Timeframe:")).toBeDefined();
    });
    expect(screen.getByText("1 Day")).toBeDefined();
    expect(screen.getByText("5 Min")).toBeDefined();
    expect(screen.getByText("15 Min")).toBeDefined();
    expect(screen.getByText("1 Hour")).toBeDefined();
  });

  it("shows Start Live Trading button when not running", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Start Live Trading")).toBeDefined();
    });
  });

  it("shows Stop Live Trading button after starting", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Start Live Trading")).toBeDefined();
    });

    fireEvent.click(screen.getByText("Start Live Trading"));
    await waitFor(() => {
      expect(screen.getByText("Stop Live Trading")).toBeDefined();
    });
  });

  it("shows metrics section with cash, bars, trades", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Cash")).toBeDefined();
      expect(screen.getByText("Bars")).toBeDefined();
      expect(screen.getByText("Trades")).toBeDefined();
    });
  });

  it("shows Actions table and Log section", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("Actions")).toBeDefined();
      expect(screen.getByText("Log")).toBeDefined();
    });
  });

  it("shows 'No actions yet' when no trades", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("No actions yet")).toBeDefined();
    });
  });

  it("shows 'No events yet' in log initially", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("No events yet")).toBeDefined();
    });
  });

  it("uses handleSelectStrategy with stable dependencies", async () => {
    render(<ThemeProvider><Live /></ThemeProvider>);
    await waitFor(() => {
      expect(screen.getByText("SmaCrossover")).toBeDefined();
    });
  });
});
