import { describe, it, expect, vi, beforeEach } from "vitest";
import { ApiClient, BacktestSocket, LiveSocket } from "../client";

const api = new ApiClient();

beforeEach(() => {
  vi.restoreAllMocks();
});

// ── Untested API Methods ──

describe("ApiClient - ML methods", () => {
  it("getMlModels fetches model list", async () => {
    const mock = [{ name: "RF_v1", version: 1, model_type: "rf", train_date: "2025-01-01", train_symbols: ["AAPL"], context_symbols: [], validation_metrics: {}, beat_baselines: false, versions: [1] }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, status: 200, json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getMlModels();
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/ml/models");
  });

  it("retrainMlModel sends POST request", async () => {
    const req = { symbols: "AAPL", years: 5, name: "test", model_types: "rf,gbt", beat_baselines: false, grid_search: false, walk_forward: 0, stacking: false, meta_labeling: false, regularize: false, prune: 0, kelly: false, auto_threshold: false, labeling: "next_bar", forecast_horizon: 1, regime_aware: false };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ status: "started", message: "ok", pid: 123 }),
    } as Response);
    const result = await api.retrainMlModel(req);
    expect(result.status).toBe("started");
    expect(fetch).toHaveBeenCalledWith("/api/ml/retrain", expect.objectContaining({ method: "POST" }));
  });

  it("getMlRetrainStatus fetches training status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ status: "running", message: "Training..." }),
    } as Response);
    const result = await api.getMlRetrainStatus("test_model");
    expect(result.status).toBe("running");
    expect(fetch).toHaveBeenCalledWith("/api/ml/retrain/status/test_model");
  });

  it("deleteMlModel sends DELETE request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: () => Promise.resolve({}) } as Response);
    await api.deleteMlModel("test_model");
    expect(fetch).toHaveBeenCalledWith(
      "/api/ml/models/test_model",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("getMlModel fetches single model", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ name: "RF_v1", version: 1 }),
    } as Response);
    await api.getMlModel("RF_v1");
    expect(fetch).toHaveBeenCalledWith("/api/ml/models/RF_v1");
  });
});

describe("ApiClient - Alpaca methods", () => {
  it("getAlpacaAccount fetches account data", async () => {
    const mock = { cash: 10000, portfolio_value: 15000, buying_power: 20000, day_pnl: 100 };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getAlpacaAccount();
    expect(result).toEqual(mock);
  });

  it("getAlpacaAccount throws on error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false, status: 503, statusText: "Service Unavailable",
    } as Response);
    await expect(api.getAlpacaAccount()).rejects.toThrow("Alpaca account not available");
  });

  it("getAlpacaPositions fetches positions", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve([]),
    } as Response);
    const result = await api.getAlpacaPositions();
    expect(result).toEqual([]);
  });

  it("getAlpacaOrders fetches orders with limit", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve([]),
    } as Response);
    await api.getAlpacaOrders(50);
    expect(fetch).toHaveBeenCalledWith("/api/alpaca/orders?limit=50");
  });

  it("getAlpacaPortfolioHistory fetches history", async () => {
    const mock = [{ timestamp: 1700000000, equity: 10000 }];
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve(mock),
    } as Response);
    const result = await api.getAlpacaPortfolioHistory("1M", "1D");
    expect(result).toEqual(mock);
    expect(fetch).toHaveBeenCalledWith("/api/alpaca/portfolio-history?period=1M&timeframe=1D");
  });
});

describe("ApiClient - Pairs methods", () => {
  it("analyzePair sends POST request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ status: "ok", pair: {} }),
    } as Response);
    const result = await api.analyzePair({ ticker_a: "AAPL", ticker_b: "MSFT" });
    expect(result.status).toBe("ok");
    expect(fetch).toHaveBeenCalledWith(
      "/api/pairs/analyze",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("rankPairs sends POST request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ status: "ok", ranked_pairs: [] }),
    } as Response);
    const result = await api.rankPairs({ pairs: [["AAPL", "MSFT"]], top_n: 5 });
    expect(result.status).toBe("ok");
  });

  it("buildHeatmap sends POST request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ status: "ok", tickers: ["AAPL", "MSFT"], matrix: {} }),
    } as Response);
    const result = await api.buildHeatmap({ tickers: ["AAPL", "MSFT"] });
    expect(result.status).toBe("ok");
  });

  it("getCorrelationData fetches correlation data", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ symbol_a: "AAPL", symbol_b: "MSFT", correlations: {}, cumulative_returns: {}, statistics: {} }),
    } as Response);
    const result = await api.getCorrelationData("AAPL", "MSFT");
    expect(result.symbol_a).toBe("AAPL");
    expect(result.symbol_b).toBe("MSFT");
  });
});

describe("ApiClient - Backtest CRUD methods", () => {
  it("deleteBacktestRun sends DELETE request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: () => Promise.resolve({}) } as Response);
    await api.deleteBacktestRun(1);
    expect(fetch).toHaveBeenCalledWith(
      "/api/backtest/runs/1",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("clearBacktestRuns sends DELETE request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: () => Promise.resolve({}) } as Response);
    await api.clearBacktestRuns();
    expect(fetch).toHaveBeenCalledWith(
      "/api/backtest/runs",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("getLiveStatus fetches live status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, json: () => Promise.resolve({ running: false, strategy_name: null, symbols: [], timeframe: null, cash: 0, equity: 0, bar_count: 0 }),
    } as Response);
    const result = await api.getLiveStatus();
    expect(result.running).toBe(false);
  });

  it("stopLive sends POST request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: () => Promise.resolve({}) } as Response);
    await api.stopLive();
    expect(fetch).toHaveBeenCalledWith(
      "/api/live/stop",
      expect.objectContaining({ method: "POST" })
    );
  });
});

// ── HTTP error handling edge cases ──

describe("ApiClient - HTTP error handling", () => {
  it("throws on HTTP 400 with status text", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false, status: 400, statusText: "Bad Request",
    } as Response);
    await expect(api.getStrategies()).rejects.toThrow("HTTP 400: Bad Request");
  });

  it("throws on HTTP 500 with status text", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false, status: 500, statusText: "Internal Server Error",
    } as Response);
    await expect(api.getStrategies()).rejects.toThrow("HTTP 500: Internal Server Error");
  });

  it("throws on network error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.getStrategies()).rejects.toThrow("Failed to fetch");
  });
});

// ── BacktestSocket edge cases ──

describe("BacktestSocket - Edge cases", () => {
  let socket: BacktestSocket;

  beforeEach(() => {
    socket = new BacktestSocket("ws://localhost/test");
  });

  it("double connect closes old WebSocket before creating new one", async () => {
    // Bug: BacktestSocket.connect() doesn't close old this.ws before
    // creating a new one, leaking the old connection.
    const closeMock = vi.fn();
    const ws1 = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: closeMock, send: vi.fn() };
    const ws2 = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };

    let callCount = 0;
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => {
      callCount++;
      return callCount === 1 ? ws1 : ws2;
    });

    // First connect
    const p1 = socket.connect({});
    (ws1 as any).onopen?.();
    await p1;

    // Second connect without closing first
    // Bug: ws1.close() should be called here but isn't
    const p2 = socket.connect({});
    (ws2 as any).onopen?.();
    await p2;

    // Bug: ws1.close() was never called — the old WebSocket is leaked
    expect(closeMock).toHaveBeenCalled();
  });

  it("send before connect completes does not throw", async () => {
    // Bug: send() uses `this.ws?.send()` which silently drops messages
    // if ws is null (not yet connected)
    const ws = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    // Send before connect resolves — ws is still null
    // This should not throw even though no message is sent
    expect(() => socket.send({ action: "test" })).not.toThrow();
  });

  it("close before connect resolves cleans up", async () => {
    const ws = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    // Start connecting
    const connectPromise = socket.connect({});

    // Close before connect resolves
    socket.close();
    expect(ws.close).toHaveBeenCalled();
  });

  it("handles JSON parse failure in onmessage gracefully", async () => {
    const onBar = vi.fn();
    const ws = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    const connectPromise = socket.connect({ onBar });
    (ws as any).onopen?.();
    await connectPromise;

    // Send invalid JSON
    // Bug: onmessage catches JSON parse errors silently (line 318-320)
    // The error is swallowed — onBar is never called
    ws.onmessage?.(new MessageEvent("message", { data: "not valid json" }));
    expect(onBar).not.toHaveBeenCalled();
  });

  it("unknown event type does not crash", async () => {
    const onBar = vi.fn();
    const ws = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    const connectPromise = socket.connect({ onBar });
    (ws as any).onopen?.();
    await connectPromise;

    // Unknown event type — switch silently drops it
    ws.onmessage?.(new MessageEvent("message", {
      data: JSON.stringify({ type: "unknown", data: "test" }),
    }));
    expect(onBar).not.toHaveBeenCalled();
  });

  it("close callback fires on server close", async () => {
    const onClose = vi.fn();
    const ws = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    socket.onClose(onClose);
    const connectPromise = socket.connect({});
    (ws as any).onopen?.();
    await connectPromise;

    // Simulate server closing the connection
    // Bug: onclose handler calls closeCallback but doesn't clean up this.ws
    // This means the socket is in a closed-but-not-null state
    const closeHandler = (ws as any).onclose;
    closeHandler?.();

    expect(onClose).toHaveBeenCalled();
  });
});

// ── LiveSocket edge cases (mirrors BacktestSocket bugs) ──

describe("LiveSocket - Edge cases", () => {
  let socket: LiveSocket;

  beforeEach(() => {
    socket = new LiveSocket("ws://localhost/live");
  });

  it("double connect leaks old WebSocket", async () => {
    // Bug: same as BacktestSocket — no guard against multiple connect() calls
    const closeMock = vi.fn();
    const ws1 = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: closeMock, send: vi.fn() };
    const ws2 = { onopen: null as null | (() => void), onmessage: null, onerror: null, close: vi.fn(), send: vi.fn() };

    let callCount = 0;
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => {
      callCount++;
      return callCount === 1 ? ws1 : ws2;
    });

    const p1 = socket.connect({});
    (ws1 as any).onopen?.();
    await p1;

    const p2 = socket.connect({});
    (ws2 as any).onopen?.();
    await p2;

    // Bug: ws1.close() was never called
    expect(closeMock).toHaveBeenCalled();
  });

  it("handles status events", async () => {
    const onStatus = vi.fn();
    const ws = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    const connectPromise = socket.connect({ onStatus });
    (ws as any).onopen?.();
    await connectPromise;

    ws.onmessage?.(new MessageEvent("message", {
      data: JSON.stringify({ type: "status", status: "running", message: "Engine started" }),
    }));

    expect(onStatus).toHaveBeenCalledWith(
      expect.objectContaining({ status: "running" })
    );
  });

  it("handles error events", async () => {
    const onError = vi.fn();
    const ws = { onopen: null as null | (() => void), onmessage: null as null | ((e: MessageEvent) => void), onerror: null, close: vi.fn(), send: vi.fn() };
    vi.spyOn(globalThis, "WebSocket" as any).mockImplementation(() => ws);

    const connectPromise = socket.connect({ onError });
    (ws as any).onopen?.();
    await connectPromise;

    ws.onmessage?.(new MessageEvent("message", {
      data: JSON.stringify({ type: "error", message: "Something went wrong" }),
    }));

    expect(onError).toHaveBeenCalledWith("Something went wrong");
  });
});
