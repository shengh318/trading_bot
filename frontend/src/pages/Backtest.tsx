import { useEffect, useRef, useState, useCallback } from "react";
import {
  api,
  type StrategyInfo,
  type BacktestRun,
  type BacktestMetrics,
  type Trade,
} from "../api/client";
import StrategySelector from "../components/StrategySelector";
import PortfolioChart from "../components/PortfolioChart";

const EMPTY_METRICS: BacktestMetrics = {
  total_return_pct: 0,
  final_equity: 0,
  sharpe_ratio: 0,
  max_drawdown_pct: 0,
  win_rate_pct: 0,
  num_trades: 0,
  profit_factor: 0,
};

export default function Backtest() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [symbol, setSymbol] = useState("AAPL");
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2024-06-01");
  const [initialCash, setInitialCash] = useState(10000);
  const [running, setRunning] = useState(false);
  const [pastRuns, setPastRuns] = useState<BacktestRun[]>([]);
  const [equityPoints, setEquityPoints] = useState<{ time: import("lightweight-charts").Time; value: number }[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const wsRef = useRef<ReturnType<typeof api.createBacktestSocket> | null>(null);

  useEffect(() => {
    api.getStrategies().then((list) => {
      setStrategies(list);
      if (list.length > 0) {
        setSelectedStrategy(list[0].name);
        const defaults: Record<string, unknown> = {};
        for (const p of list[0].params) {
          defaults[p.name] = p.default;
        }
        setParams(defaults);
      }
    });
    api.getBacktestRuns().then(setPastRuns);
  }, []);

  const handleSelectStrategy = useCallback(
    (name: string) => {
      setSelectedStrategy(name);
      const s = strategies.find((st) => st.name === name);
      if (s) {
        const defaults: Record<string, unknown> = {};
        for (const p of s.params) {
          defaults[p.name] = p.default;
        }
        setParams(defaults);
      }
    },
    [strategies],
  );

  const addLog = (msg: string) => setLog((prev) => [...prev, msg]);

  const runBacktest = useCallback(async () => {
    setRunning(true);
    setEquityPoints([]);
    setTrades([]);
    setMetrics(null);
    setLog([]);

    const ws = api.createBacktestSocket();
    wsRef.current = ws;

    try {
      await ws.connect({
        onBar: (event) => {
          setEquityPoints((prev) => [
            ...prev,
            {
              time: (new Date(event.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
              value: event.equity,
            },
          ]);
          if (event.trade) {
            setTrades((prev) => [...prev, event.trade!]);
            addLog(
              `${event.trade.side.toUpperCase()} ${event.trade.qty} ${event.trade.symbol} @ $${event.trade.price}`,
            );
          }
        },
        onComplete: (event) => {
          setMetrics((event.metrics?.metrics as BacktestMetrics) ?? EMPTY_METRICS);
          setRunning(false);
          addLog("Backtest complete!");
          api.getBacktestRuns().then(setPastRuns);
        },
        onError: (message) => {
          addLog(`Error: ${message}`);
          setRunning(false);
        },
      });

      ws.send({
        action: "run",
        strategy_name: selectedStrategy,
        symbol,
        start_date: startDate,
        end_date: endDate,
        initial_cash: initialCash,
        parameters: params,
      });
    } catch {
      addLog("Failed to connect to WebSocket");
      setRunning(false);
    }
  }, [selectedStrategy, symbol, startDate, endDate, initialCash, params]);

  const replayRun = useCallback(async (runId: number) => {
    setRunning(true);
    setEquityPoints([]);
    setTrades([]);
    setMetrics(null);
    setLog([]);

    const ws = api.createBacktestSocket();
    wsRef.current = ws;

    try {
      await ws.connect({
        onBar: (event) => {
          setEquityPoints((prev) => [
            ...prev,
            {
              time: (new Date(event.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
              value: event.equity,
            },
          ]);
          if (event.trade) {
            setTrades((prev) => [...prev, event.trade!]);
          }
        },
        onComplete: () => {
          setRunning(false);
          addLog("Replay complete!");
        },
        onError: (message) => {
          addLog(`Error: ${message}`);
          setRunning(false);
        },
      });

      ws.send({ action: "replay", run_id: runId });
    } catch {
      addLog("Failed to connect to WebSocket");
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const metricStyle: React.CSSProperties = {
    padding: "8px 16px",
    background: "#f8f9fa",
    borderRadius: 6,
    textAlign: "center",
    fontSize: 13,
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 24 }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ margin: "0 0 12px" }}>Run Backtest</h3>
          <StrategySelector
            strategies={strategies}
            selected={selectedStrategy}
            params={params}
            onSelect={handleSelectStrategy}
            onParamChange={(name, value) =>
              setParams((prev) => ({ ...prev, [name]: value }))
            }
          />
          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>Symbol:</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              style={{ padding: "4px 8px", fontSize: 14, width: 100 }}
            />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>Start:</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ padding: "4px 8px", fontSize: 14 }}
            />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>End:</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ padding: "4px 8px", fontSize: 14 }}
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>
              Initial Cash:
            </label>
            <input
              type="number"
              value={initialCash}
              onChange={(e) => setInitialCash(Number(e.target.value))}
              style={{ padding: "4px 8px", fontSize: 14, width: 120 }}
            />
          </div>
          <button
            onClick={runBacktest}
            disabled={running}
            style={{
              padding: "8px 24px",
              background: running ? "#ccc" : "#1a73e8",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: running ? "not-allowed" : "pointer",
              fontWeight: 600,
            }}
          >
            {running ? "Running..." : "Run Backtest"}
          </button>

          <div style={{ marginTop: 24 }}>
            <h4 style={{ margin: "0 0 8px" }}>Previous Runs</h4>
            {pastRuns.length === 0 ? (
              <p style={{ fontSize: 13, color: "#777" }}>No runs yet</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>ID</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Strategy</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Symbol</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Return</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}></th>
                  </tr>
                </thead>
                <tbody>
                  {pastRuns.map((r) => (
                    <tr key={r.id}>
                      <td style={{ padding: "4px 8px" }}>{r.id}</td>
                      <td style={{ padding: "4px 8px" }}>{r.strategy_name}</td>
                      <td style={{ padding: "4px 8px" }}>{r.symbol}</td>
                      <td style={{ padding: "4px 8px" }}>
                        {r.metrics
                          ? `${r.metrics.total_return_pct >= 0 ? "+" : ""}${r.metrics.total_return_pct.toFixed(2)}%`
                          : "-"}
                      </td>
                      <td style={{ padding: "4px 8px" }}>
                        <button
                          onClick={() => replayRun(r.id)}
                          disabled={running}
                          style={{
                            padding: "2px 10px",
                            background: "#e8eaed",
                            border: "none",
                            borderRadius: 4,
                            cursor: "pointer",
                            fontSize: 12,
                          }}
                        >
                          Replay
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div style={{ flex: 2 }}>
          <PortfolioChart data={equityPoints} />

          {metrics && (
            <div
              style={{
                display: "flex",
                gap: 12,
                flexWrap: "wrap",
                marginBottom: 16,
              }}
            >
              <div style={metricStyle}>
                <strong>Return</strong>
                <div style={{ color: metrics.total_return_pct >= 0 ? "#0b8043" : "#c5221f" }}>
                  {metrics.total_return_pct >= 0 ? "+" : ""}
                  {metrics.total_return_pct.toFixed(2)}%
                </div>
              </div>
              <div style={metricStyle}>
                <strong>Final Equity</strong>
                <div>${metrics.final_equity.toFixed(2)}</div>
              </div>
              <div style={metricStyle}>
                <strong>Sharpe</strong>
                <div>{metrics.sharpe_ratio.toFixed(2)}</div>
              </div>
              <div style={metricStyle}>
                <strong>Max DD</strong>
                <div style={{ color: "#c5221f" }}>{metrics.max_drawdown_pct.toFixed(2)}%</div>
              </div>
              <div style={metricStyle}>
                <strong>Win Rate</strong>
                <div>{metrics.win_rate_pct.toFixed(1)}%</div>
              </div>
              <div style={metricStyle}>
                <strong>Trades</strong>
                <div>{metrics.num_trades}</div>
              </div>
              <div style={metricStyle}>
                <strong>Profit Factor</strong>
                <div>
                  {metrics.profit_factor === Infinity
                    ? "∞"
                    : metrics.profit_factor.toFixed(2)}
                </div>
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: 24 }}>
            <div style={{ flex: 1 }}>
              <h4 style={{ margin: "0 0 8px" }}>Trades</h4>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Bar</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Side</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Qty</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>Price</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: "1px solid #ddd" }}>P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.length === 0 ? (
                    <tr>
                      <td colSpan={5} style={{ padding: "4px 8px", color: "#777" }}>
                        No trades
                      </td>
                    </tr>
                  ) : (
                    trades.map((t, i) => (
                      <tr key={i}>
                        <td style={{ padding: "4px 8px" }}>{t.bar_index}</td>
                        <td
                          style={{
                            padding: "4px 8px",
                            color: t.side === "buy" ? "#0b8043" : "#c5221f",
                          }}
                        >
                          {t.side}
                        </td>
                        <td style={{ padding: "4px 8px" }}>{t.qty}</td>
                        <td style={{ padding: "4px 8px" }}>${t.price.toFixed(2)}</td>
                        <td
                          style={{
                            padding: "4px 8px",
                            color: t.pnl != null ? (t.pnl >= 0 ? "#0b8043" : "#c5221f") : undefined,
                          }}
                        >
                          {t.pnl != null ? `$${t.pnl.toFixed(2)}` : "-"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            <div style={{ flex: 1 }}>
              <h4 style={{ margin: "0 0 8px" }}>Log</h4>
              <div
                style={{
                  maxHeight: 200,
                  overflowY: "auto",
                  background: "#1e1e1e",
                  color: "#d4d4d4",
                  padding: 8,
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily: "monospace",
                }}
              >
                {log.length === 0 ? (
                  <span style={{ color: "#777" }}>No events yet</span>
                ) : (
                  log.map((line, i) => <div key={i}>{line}</div>)
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
