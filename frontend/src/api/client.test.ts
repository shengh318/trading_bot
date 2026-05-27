import { describe, it, expect, vi, beforeEach } from "vitest";
import { ApiClient, BacktestSocket, type WsEvent } from "./client";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("ApiClient", () => {
  const api = new ApiClient();

  it("fetches strategies", async () => {
    const mock = [{ name: "SmaCrossover", description: "Test strat", params: [] }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getStrategies();
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/strategies");
  });

  it("fetches account summary", async () => {
    const mock = { cash: 1000, portfolio_value: 5000, buying_power: 2000, day_pnl: 100 };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getAccountSummary();
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/portfolio/summary");
  });

  it("fetches equity curve with limit", async () => {
    const mock = [{ timestamp: "2024-01-01", total_equity: 10000, cash: 5000 }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getEquityCurve(100);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/portfolio/equity-curve?limit=100");
  });

  it("fetches positions", async () => {
    const mock = [{ symbol: "AAPL", qty: 10, avg_entry_price: 150, current_price: 155, unrealized_pl: 50, market_value: 1550 }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getPositions();
    expect(result).toEqual(mock);
  });

  it("fetches orders", async () => {
    const mock = [{ id: "1", symbol: "AAPL", side: "buy", qty: 10, filled_qty: 10, filled_avg_price: 150, status: "filled", type: "market", created_at: "2024-01-01", updated_at: "2024-01-01" }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getOrders(50);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/orders?limit=50");
  });

  it("fetches backtest runs", async () => {
    const mock = [{ id: 1, strategy_name: "SmaCrossover", symbol: "AAPL", start_date: "2024-01-01", end_date: "2024-06-01", initial_cash: 10000, metrics: null, created_at: "2024-01-01" }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getBacktestRuns();
    expect(result).toEqual(mock);
  });

  it("fetches a single backtest run", async () => {
    const mock = { id: 1, strategy_name: "SmaCrossover", symbol: "AAPL", start_date: "2024-01-01", end_date: "2024-06-01", initial_cash: 10000, metrics: { total_return_pct: 10, final_equity: 11000, sharpe_ratio: 1.5, max_drawdown_pct: -5, win_rate_pct: 60, num_trades: 10, profit_factor: 1.2 }, created_at: "2024-01-01" };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getBacktestRun(1);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/backtest/runs/1");
  });

  it("fetches backtest trades", async () => {
    const mock = [{ bar_index: 0, timestamp: "2024-01-01", symbol: "AAPL", side: "buy", qty: 10, price: 150, pnl: null }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getBacktestTrades(1);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/backtest/runs/1/trades");
  });

  it("fetches backtest equity snapshots", async () => {
    const mock = [{ bar_index: 0, timestamp: "2024-01-01", equity: 10000, cash: 5000 }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getBacktestEquity(1);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/backtest/runs/1/equity");
  });

  it("runs backtest via POST", async () => {
    const mock = { id: 2, strategy_name: "SmaCrossover", symbol: "AAPL", start_date: "2024-01-01", end_date: "2024-06-01", initial_cash: 10000, metrics: null, created_at: "2024-01-01" };
    const req = { strategy_name: "SmaCrossover", symbol: "AAPL", start_date: "2024-01-01", end_date: "2024-06-01", initial_cash: 10000 };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.runBacktest(req);
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/backtest/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  });
});

describe("BacktestSocket", () => {
  let socket: BacktestSocket;

  beforeEach(() => {
    vi.restoreAllMocks();
    socket = new BacktestSocket("ws://localhost/test");
  });

  it("connect resolves on open", async () => {
    const mockWs = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({});
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;
  });

  it("calls onBar on bar event", async () => {
    const onBar = vi.fn();
    const mockWs = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({ onBar });
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;

    const event: WsEvent = {
      type: "bar",
      bar_index: 0,
      timestamp: "2024-01-01",
      equity: 10000,
      cash: 5000,
      trades: [],
    };
    mockWs.onmessage?.(new MessageEvent("message", { data: JSON.stringify(event) }));
    expect(onBar).toHaveBeenCalledWith(event);
  });

  it("calls onComplete on complete event", async () => {
    const onComplete = vi.fn();
    const mockWs = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({ onComplete });
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;

    const event: WsEvent = {
      type: "complete",
      run_id: 1,
    };
    mockWs.onmessage?.(new MessageEvent("message", { data: JSON.stringify(event) }));
    expect(onComplete).toHaveBeenCalledWith(event);
  });

  it("calls onError on error event", async () => {
    const onError = vi.fn();
    const mockWs = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({ onError });
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;

    mockWs.onmessage?.(new MessageEvent("message", { data: JSON.stringify({ type: "error", message: "test error" }) }));
    expect(onError).toHaveBeenCalledWith("test error");
  });

  it("sends JSON via WebSocket", async () => {
    const send = vi.fn();
    const mockWs = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({});
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;

    socket.send({ action: "run", symbol: "AAPL" });
    expect(send).toHaveBeenCalledWith(JSON.stringify({ action: "run", symbol: "AAPL" }));
  });

  it("closes WebSocket", async () => {
    const close = vi.fn();
    const mockWs = { onopen: null as null | (() => void), onmessage: null, onerror: null, close, send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(
      () => mockWs as unknown as WebSocket,
    );
    const connectPromise = socket.connect({});
    setTimeout(() => mockWs.onopen?.(), 0);
    await connectPromise;

    socket.close();
    expect(close).toHaveBeenCalled();
  });
});
