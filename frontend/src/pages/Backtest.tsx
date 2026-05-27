import { useEffect, useRef, useState, useCallback } from "react";
import {
  api,
  type StrategyInfo,
  type BacktestRun,
  type BacktestMetrics,
  type Trade,
} from "../api/client";
import { useTheme } from "../theme/ThemeContext";
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
  const { colors } = useTheme();
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [symbol, setSymbol] = useState("AAPL");
  const [startDate, setStartDate] = useState("2024-01-01");
  const [initialCash, setInitialCash] = useState("100");
  const [running, setRunning] = useState(false);
  const [pastRuns, setPastRuns] = useState<BacktestRun[]>([]);
  const [equityPoints, setEquityPoints] = useState<{ time: import("lightweight-charts").Time; value: number }[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const wsRef = useRef<ReturnType<typeof api.createBacktestSocket> | null>(null);

  const markers = trades.map((t) => ({
    time: (new Date(t.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
    side: t.side as "buy" | "sell",
  }));

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

  const clearRuns = async () => {
    await api.clearBacktestRuns();
    setPastRuns([]);
  };

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
          if (event.dividend) {
            addLog(
              `DIVIDEND $${event.dividend.dividend.toFixed(2)} received`,
            );
          }
        },
        onComplete: (event) => {
          setMetrics(event.metrics ?? EMPTY_METRICS);
          setRunning(false);
          addLog("Simulation complete!");
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
        initial_cash: Number(initialCash),
        parameters: params,
      });
    } catch {
      addLog("Failed to connect to WebSocket");
      setRunning(false);
    }
  }, [selectedStrategy, symbol, startDate, initialCash, params]);

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
        onComplete: (event) => {
          setMetrics(event.metrics ?? EMPTY_METRICS);
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
    background: colors.surface,
    borderRadius: 6,
    textAlign: "center",
    fontSize: 13,
  };

  const inputStyle: React.CSSProperties = {
    padding: "4px 8px",
    fontSize: 14,
    background: colors.inputBg,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 24 }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ margin: "0 0 12px" }}>Run Simulation</h3>
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
              style={{ ...inputStyle, width: 100 }}
            />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>Start:</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={inputStyle}
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>
              Starting Amount ($):
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={initialCash}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "" || /^\d+(\.\d*)?$/.test(v)) setInitialCash(v);
              }}
              style={{ ...inputStyle, width: 120 }}
            />
          </div>
          <button
            onClick={runBacktest}
            disabled={running}
            style={{
              padding: "8px 24px",
              background: running ? colors.border : colors.primary,
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: running ? "not-allowed" : "pointer",
              fontWeight: 600,
            }}
          >
            {running ? "Running..." : "Run Simulation"}
          </button>

          <div style={{ marginTop: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <h4 style={{ margin: 0 }}>Previous Runs</h4>
              {pastRuns.length > 0 && (
                <button
                  onClick={clearRuns}
                  style={{
                    padding: "2px 10px",
                    background: colors.negative,
                    color: "#fff",
                    border: "none",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  Clear All
                </button>
              )}
            </div>
            {pastRuns.length === 0 ? (
              <p style={{ fontSize: 13, color: colors.textMuted }}>No runs yet</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>ID</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Strategy</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Symbol</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Return</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}></th>
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
                            background: colors.tabInactive,
                            color: colors.text,
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
          <PortfolioChart data={equityPoints} markers={equityPoints.length > 0 ? markers : undefined} initialCash={Number(initialCash)} />

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
                <div style={{ color: metrics.total_return_pct >= 0 ? colors.positive : colors.negative }}>
                  {metrics.total_return_pct >= 0 ? "+" : ""}
                  {metrics.total_return_pct.toFixed(2)}%
                </div>
              </div>
              <div style={metricStyle}>
                <strong>Final Value</strong>
                <div>${metrics.final_equity.toFixed(2)}</div>
              </div>
              <div style={metricStyle}>
                <strong>Risk Score</strong>
                <div>{metrics.sharpe_ratio.toFixed(2)}</div>
              </div>
              <div style={metricStyle}>
                <strong>Drop</strong>
                <div style={{ color: colors.negative }}>{metrics.max_drawdown_pct.toFixed(2)}%</div>
              </div>
              <div style={metricStyle}>
                <strong>Win %</strong>
                <div>{metrics.win_rate_pct.toFixed(1)}%</div>
              </div>
              <div style={metricStyle}>
                <strong>Trades</strong>
                <div>{metrics.num_trades}</div>
              </div>
              <div style={metricStyle}>
                <strong>Profit Ratio</strong>
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
              <h4 style={{ margin: "0 0 8px" }}>Actions</h4>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Bar</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Action</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Qty</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Price</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Profit/Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.length === 0 ? (
                    <tr>
                      <td colSpan={5} style={{ padding: "4px 8px", color: colors.textMuted }}>
                        No actions yet
                      </td>
                    </tr>
                  ) : (
                    trades.map((t, i) => (
                      <tr key={i}>
                        <td style={{ padding: "4px 8px" }}>{t.bar_index}</td>
                        <td
                          style={{
                            padding: "4px 8px",
                            color: t.side === "buy" ? colors.positive : colors.negative,
                          }}
                        >
                          {t.side}
                        </td>
                        <td style={{ padding: "4px 8px" }}>{t.qty}</td>
                        <td style={{ padding: "4px 8px" }}>${t.price.toFixed(2)}</td>
                        <td
                          style={{
                            padding: "4px 8px",
                            color: t.pnl != null ? (t.pnl >= 0 ? colors.positive : colors.negative) : undefined,
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
                  background: colors.logBg,
                  color: colors.logText,
                  padding: 8,
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily: "monospace",
                }}
              >
                {log.length === 0 ? (
                  <span style={{ color: colors.textMuted }}>No events yet</span>
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
