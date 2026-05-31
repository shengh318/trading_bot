import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  api,
  type StrategyInfo,
  type Trade,
} from "../api/client";
import { useTheme } from "../theme/ThemeContext";
import StrategySelector from "../components/StrategySelector";
import PortfolioChart from "../components/PortfolioChart";

const AVAILABLE_SYMBOLS = ["NVDA", "AMD", "VOO", "SPY", "META"];

export default function Live() {
  const { colors } = useTheme();
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([...AVAILABLE_SYMBOLS]);
  const [timeframe, setTimeframe] = useState("1Day");
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [equityPoints, setEquityPoints] = useState<{ time: import("lightweight-charts").Time; value: number }[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [cash, setCash] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const wsRef = useRef<ReturnType<typeof api.createLiveSocket> | null>(null);

  const markers = useMemo(
    () =>
      trades
        .filter((t): t is Trade & { side: "buy" | "sell" } => t.side === "buy" || t.side === "sell")
        .map((t) => ({
          time: (new Date(t.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
          side: t.side,
        })),
    [trades],
  );

  const toggleSymbol = useCallback((sym: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  }, []);

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
    }).catch((e: unknown) => {
      addLog(`Failed to load strategies: ${e instanceof Error ? e.message : String(e)}`);
    });
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

  const addLog = useCallback((msg: string) => setLog((prev) => [...prev, msg]), []);

  const equityRef = useRef<{ time: import("lightweight-charts").Time; value: number }[]>([]);

  const startLive = useCallback(async () => {
    setRunning(true);
    setStatusMessage("Connecting...");
    setEquityPoints([]);
    setTrades([]);
    setLog([]);
    setCash(0);
    equityRef.current = [];

    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = api.createLiveSocket();
    wsRef.current = ws;

    try {
      await ws.connect({
        onBar: (event) => {
          const point = {
            time: (new Date(event.timestamp).getTime() / 1000) as unknown as import("lightweight-charts").Time,
            value: event.equity,
          };
          equityRef.current.push(point);
          setEquityPoints([...equityRef.current]);
          setCash(event.cash);
          for (const trade of event.trades) {
            setTrades((prev) => [...prev, trade]);
            addLog(
              `${trade.side.toUpperCase()} ${Number.isInteger(trade.qty) ? trade.qty : trade.qty.toFixed(4)} ${trade.symbol} @ $${trade.price.toFixed(2)}`,
            );
          }
        },
        onStatus: (event) => {
          setStatusMessage(event.message);
          if (event.status === "stopped") {
            setRunning(false);
          }
          addLog(`[${event.status}] ${event.message}`);
        },
        onError: (message) => {
          addLog(`Error: ${message}`);
          setRunning(false);
        },
      });

      ws.onClose(() => setRunning(false));

      ws.send({
        action: "start",
        strategy_name: selectedStrategy,
        symbols: selectedSymbols,
        parameters: params,
        timeframe,
      });
    } catch {
      addLog("Failed to connect to WebSocket");
      setRunning(false);
    }
  }, [selectedStrategy, selectedSymbols, params, timeframe]);

  const stopLive = useCallback(async () => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.send({ action: "stop" });
      ws.close();
    }
    await api.stopLive();
    setRunning(false);
    setStatusMessage("Stopped");
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const inputStyle: React.CSSProperties = {
    padding: "4px 8px",
    fontSize: 14,
    background: colors.inputBg,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
  };

  const metricStyle: React.CSSProperties = {
    padding: "8px 16px",
    background: colors.surface,
    borderRadius: 6,
    textAlign: "center",
    fontSize: 13,
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 24 }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ margin: "0 0 12px" }}>Live Trading</h3>
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
            <label style={{ marginRight: 8, fontWeight: 600 }}>Symbols:</label>
            {AVAILABLE_SYMBOLS.map((sym) => (
              <label key={sym} style={{ marginRight: 12, cursor: "pointer", fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={selectedSymbols.includes(sym)}
                  onChange={() => toggleSymbol(sym)}
                  style={{ marginRight: 4 }}
                />
                {sym}
              </label>
            ))}
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>Timeframe:</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              style={{ ...inputStyle, width: 100 }}
            >
              <option value="1Day">1 Day</option>
              <option value="5Min">5 Min</option>
              <option value="15Min">15 Min</option>
              <option value="1Hour">1 Hour</option>
            </select>
          </div>
          {!running ? (
            <button
              onClick={startLive}
              style={{
                padding: "8px 24px",
                background: colors.primary,
                color: "#fff",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Start Live Trading
            </button>
          ) : (
            <button
              onClick={stopLive}
              style={{
                padding: "8px 24px",
                background: colors.negative,
                color: "#fff",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Stop Live Trading
            </button>
          )}
          <div style={{ marginTop: 8, fontSize: 13, color: colors.textMuted }}>
            {statusMessage}
          </div>
        </div>

        <div style={{ flex: 2 }}>
          <div style={{ position: "relative" }}>
            <PortfolioChart data={equityPoints} markers={equityPoints.length > 0 ? markers : undefined} />
            {equityPoints.length <= 1 && running && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  pointerEvents: "none",
                  fontSize: 15,
                  color: colors.textMuted,
                }}
              >
                {statusMessage || "Waiting for market data…"}
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
            <div style={metricStyle}>
              <strong>Cash</strong>
              <div>${cash.toFixed(2)}</div>
            </div>
            <div style={metricStyle}>
              <strong>Bars</strong>
              <div>{equityPoints.length}</div>
            </div>
            <div style={metricStyle}>
              <strong>Trades</strong>
              <div>{trades.length}</div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 24 }}>
            <div style={{ flex: 1 }}>
              <h4 style={{ margin: "0 0 8px" }}>Actions</h4>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Symbol</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Action</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Qty</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>Price</th>
                    <th style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${colors.tableBorder}` }}>P/L</th>
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
                        <td style={{ padding: "4px 8px" }}>{t.symbol}</td>
                        <td style={{ padding: "4px 8px", color: t.side === "buy" ? colors.positive : colors.negative }}>
                          {t.side}
                        </td>
                        <td style={{ padding: "4px 8px" }}>{t.qty}</td>
                        <td style={{ padding: "4px 8px" }}>${t.price.toFixed(2)}</td>
                        <td style={{ padding: "4px 8px", color: t.pnl != null ? (t.pnl >= 0 ? colors.positive : colors.negative) : undefined }}>
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
