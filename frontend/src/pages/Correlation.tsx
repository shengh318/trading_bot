import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, type ISeriesApi, type Time } from "lightweight-charts";
import { api, type CorrelationData } from "../api/client";
import { useTheme } from "../theme/ThemeContext";

const LINE_COLORS = ["#2962FF", "#FF6D00", "#00C853", "#D50000", "#AA00FF"];
const WINDOW_OPTIONS = [20, 60, 120];

export default function Correlation() {
  const { colors } = useTheme();

  const [symbolA, setSymbolA] = useState("NVDA");
  const [symbolB, setSymbolB] = useState("SPY");
  const [yearsStr, setYearsStr] = useState("5");
  const [selectedWindows, setSelectedWindows] = useState<number[]>([20, 60]);
  const [data, setData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const containerRef = useRef<HTMLDivElement>(null);

  const toggleWindow = (w: number) => {
    setSelectedWindows((prev) =>
      prev.includes(w) ? prev.filter((x) => x !== w) : [...prev, w].sort((a, b) => a - b)
    );
  };

  const loadData = async () => {
    setLoading(true);
    setError("");
    const yearsNum = Math.max(1, parseInt(yearsStr.replace(/\D/g, "") || "5", 10));
    try {
      const result = await api.getCorrelationData(
        symbolA.toUpperCase(),
        symbolB.toUpperCase(),
        yearsNum,
        selectedWindows.join(",")
      );
      setData(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      console.error("Correlation load error:", msg);
    } finally {
      setLoading(false);
    }
  };

  // Auto-load on mount
  useEffect(() => {
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Build chart whenever data or selectedWindows changes
  useEffect(() => {
    if (!containerRef.current || !data) return;

    const chart = createChart(containerRef.current, {
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: colors.bg },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.chart.gridColor },
        horzLines: { color: colors.chart.gridColor },
      },
      timeScale: {
        borderColor: colors.chart.borderColor,
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: colors.chart.borderColor,
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
      crosshair: {
        mode: 0,
      },
    });

    const seriesList: ISeriesApi<"Line">[] = [];

    const keys = Object.keys(data.correlations).filter((k) =>
      selectedWindows.includes(Number(k))
    );

    keys.forEach((key, idx) => {
      const points = data.correlations[key].map((p) => ({
        time: p.time as unknown as Time,
        value: p.value,
      }));

      if (idx === 0 && points.length > 0) {
        const zeroData = points.map((p) => ({ time: p.time, value: 0 }));
        const refSeries = chart.addLineSeries({
          color: colors.chart.gridColor,
          lineWidth: 1,
          lineStyle: 2,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        refSeries.setData(zeroData);
        seriesList.push(refSeries);
      }

      const lineSeries = chart.addLineSeries({
        color: LINE_COLORS[(idx + 1) % LINE_COLORS.length],
        lineWidth: 2,
        lastValueVisible: true,
        priceLineVisible: false,
        title: `${key}-day`,
      });
      lineSeries.setData(points);
      seriesList.push(lineSeries);
    });

    chart.timeScale().fitContent();

    const handleResize = () => {
      chart.applyOptions({ width: containerRef.current!.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, selectedWindows, colors]);

  const statStyle: React.CSSProperties = {
    padding: "8px 12px",
    background: colors.surface,
    borderRadius: 6,
    minWidth: 100,
    textAlign: "center",
  };
  const statLabel = { fontSize: 11, color: colors.text, opacity: 0.7 };
  const statValue = { fontSize: 15, fontWeight: 700, color: colors.text };

  return (
    <div>
      {/* Controls */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "flex-end",
          flexWrap: "wrap",
          marginBottom: 16,
        }}
      >
        <label style={{ fontSize: 12, color: colors.text }}>
          Symbol A
          <input
            value={symbolA}
            onChange={(e) => setSymbolA(e.target.value)}
            style={{
              display: "block",
              marginTop: 4,
              padding: "6px 10px",
              fontSize: 14,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              background: colors.surface,
              color: colors.text,
              width: 90,
            }}
          />
        </label>
        <label style={{ fontSize: 12, color: colors.text }}>
          Symbol B
          <input
            value={symbolB}
            onChange={(e) => setSymbolB(e.target.value)}
            style={{
              display: "block",
              marginTop: 4,
              padding: "6px 10px",
              fontSize: 14,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              background: colors.surface,
              color: colors.text,
              width: 90,
            }}
          />
        </label>
        <label style={{ fontSize: 12, color: colors.text }}>
          Years
          <input
            type="text"
            inputMode="numeric"
            value={yearsStr}
            onChange={(e) => setYearsStr(e.target.value.replace(/\D/g, ""))}
            style={{
              display: "block",
              marginTop: 4,
              padding: "6px 10px",
              fontSize: 14,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              background: colors.surface,
              color: colors.text,
              width: 60,
            }}
          />
        </label>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: colors.text }}>Windows:</span>
          {WINDOW_OPTIONS.map((w) => (
            <label
              key={w}
              style={{
                fontSize: 13,
                color: colors.text,
                display: "flex",
                alignItems: "center",
                gap: 4,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={selectedWindows.includes(w)}
                onChange={() => toggleWindow(w)}
              />
              {w}
            </label>
          ))}
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          style={{
            padding: "8px 20px",
            background: colors.primary,
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "not-allowed" : "pointer",
            fontWeight: 600,
            fontSize: 14,
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "Loading…" : "Load Data"}
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: 12,
            background: colors.negative + "18",
            color: colors.negative,
            borderRadius: 6,
            marginBottom: 16,
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      {/* Chart */}
      <div ref={containerRef} style={{ width: "100%", marginBottom: 16, minHeight: 400 }} />
      {loading && !data && (
        <div style={{ textAlign: "center", color: colors.textMuted, fontSize: 13, padding: 8 }}>
          Loading data…
        </div>
      )}

      {/* Statistics */}
      {data && (
        <div
          style={{
            display: "flex",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 16,
          }}
        >
          <div style={statStyle}>
            <div style={statLabel}>Pearson r</div>
            <div style={statValue}>{data.statistics.pearson_r}</div>
            <div style={{ ...statLabel, fontSize: 10 }}>
              p={data.statistics.pearson_p}
            </div>
          </div>
          <div style={statStyle}>
            <div style={statLabel}>Spearman ρ</div>
            <div style={statValue}>{data.statistics.spearman_r}</div>
            <div style={{ ...statLabel, fontSize: 10 }}>
              p={data.statistics.spearman_p}
            </div>
          </div>
          <div style={statStyle}>
            <div style={statLabel}>Kendall τ</div>
            <div style={statValue}>{data.statistics.kendall_tau}</div>
            <div style={{ ...statLabel, fontSize: 10 }}>
              p={data.statistics.kendall_p}
            </div>
          </div>
          <div style={statStyle}>
            <div style={statLabel}>Rolling Corr Std</div>
            <div style={statValue}>{data.statistics.rolling_corr_std}</div>
          </div>
          <div style={statStyle}>
            <div style={statLabel}>OOS Corr Drop</div>
            <div style={statValue}>{data.statistics.oos_corr_drop}</div>
          </div>
        </div>
      )}

      {/* Legend */}
      {data && selectedWindows.length > 0 && (
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 13, color: colors.text }}>
          {selectedWindows.map((w, idx) => (
            <div key={w} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 12,
                  height: 3,
                  borderRadius: 2,
                  background: LINE_COLORS[(idx + 1) % LINE_COLORS.length],
                  display: "inline-block",
                }}
              />
              {w}-day correlation ({symbolA} vs {symbolB})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
