import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import { api } from "../api/client";
import type { PairData, PairAnalysisResponse } from "../api/client";
import { useTheme } from "../theme/ThemeContext";

type Section = "analyze" | "rank" | "heatmap";

function fmt(x: number | null | undefined, d = 4): string {
  if (x === null || x === undefined || !isFinite(x)) return "—";
  return x.toFixed(d);
}

function pval(p: number | undefined | null): string {
  if (p === null || p === undefined) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(4);
}

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  const { colors } = useTheme();
  return (
    <div style={{ padding: "8px 12px", background: colors.surface, borderRadius: 6, minWidth: 80, textAlign: "center", borderLeft: color ? `3px solid ${color}` : undefined }}>
      <div style={{ fontSize: 11, color: colors.textMuted }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, color: color || colors.text }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: colors.textMuted }}>{sub}</div>}
    </div>
  );
}

function RegimeBadge({ regime }: { regime: string }) {
  const colorMap: Record<string, string> = { mean_reverting: "#4CAF50", trending: "#F44336", volatile: "#FF9800", choppy: "#9E9E9E" };
  const bg = colorMap[regime] || "#9E9E9E";
  return <span style={{ padding: "2px 10px", borderRadius: 12, fontSize: 12, fontWeight: 600, background: bg + "22", color: bg }}>{regime.replace("_", " ").toUpperCase()}</span>;
}

function useChart(
  containerRef: HTMLDivElement | null,
  data: { time: number; value: number }[],
  color: string,
  options?: { zeroLine?: boolean; height?: number }
) {
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const { colors } = useTheme();

  useEffect(() => {
    if (!containerRef || data.length === 0) return;
    const h = options?.height || 250;
    const chart = createChart(containerRef, {
      height: h,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text },
      grid: { vertLines: { color: colors.chart.gridColor }, horzLines: { color: colors.chart.gridColor } },
      timeScale: { borderColor: colors.chart.borderColor, timeVisible: false },
      rightPriceScale: { borderColor: colors.chart.borderColor, scaleMargins: { top: 0.05, bottom: 0.05 } },
      crosshair: { mode: 0 },
    });
    chartRef.current = chart;

    if (options?.zeroLine !== false && data.length > 0) {
      chart.addLineSeries({ color: colors.chart.gridColor, lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false })
        .setData(data.map(p => ({ time: p.time as Time, value: 0 })));
    }

    const series = chart.addLineSeries({ color, lineWidth: 2, lastValueVisible: true, priceLineVisible: false });
    series.setData(data.map(p => ({ time: p.time as Time, value: p.value })));
    seriesRef.current = series;
    chart.timeScale().fitContent();

    const handleResize = () => chart.applyOptions({ height: containerRef.clientHeight || h });
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [containerRef, data, color, options?.height, options?.zeroLine, colors]);
}

function ChartContainer({ data, color, zeroLine, height = 250, label }: { data: { time: number; value: number }[]; color: string; zeroLine?: boolean; height?: number; label?: string }) {
  const { colors } = useTheme();
  const [ref, setRef] = useState<HTMLDivElement | null>(null);
  useChart(ref, data, color, { zeroLine, height });

  if (data.length === 0) {
    return (
      <div ref={setRef} style={{ width: "100%", height, background: colors.surface, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", color: colors.textMuted, fontSize: 13 }}>
        No data
      </div>
    );
  }

  return (
    <div>
      {label && <div style={{ fontSize: 13, fontWeight: 600, color: colors.text, marginBottom: 6 }}>{label}</div>}
      <div ref={setRef} style={{ width: "100%", height, borderRadius: 6, overflow: "hidden" }} />
    </div>
  );
}

function CollapsibleSection({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const { colors } = useTheme();
  const [open, setOpen] = useState(defaultOpen ?? true);
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        onClick={() => setOpen(!open)}
        style={{ cursor: "pointer", fontSize: 14, fontWeight: 600, color: colors.text, marginBottom: open ? 8 : 0, userSelect: "none", display: "flex", alignItems: "center", gap: 6 }}
      >
        <span style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform 0.15s", fontSize: 11 }}>▶</span>
        {title}
      </div>
      {open && children}
    </div>
  );
}

export default function Pairs() {
  const { colors } = useTheme();
  const [section, setSection] = useState<Section>("analyze");
  const [helpOpen, setHelpOpen] = useState(false);

  const inputStyle: React.CSSProperties = { display: "block", marginTop: 4, padding: "6px 10px", fontSize: 14, border: `1px solid ${colors.border}`, borderRadius: 4, background: colors.surface, color: colors.text, width: 90 };
  const btnStyle = (loading: boolean): React.CSSProperties => ({ padding: "8px 20px", background: colors.primary, color: "#fff", border: "none", borderRadius: 6, cursor: loading ? "not-allowed" : "pointer", fontWeight: 600, fontSize: 14, opacity: loading ? 0.6 : 1, alignSelf: "flex-end" });

  const [tickerA, setTickerA] = useState("XOM");
  const [tickerB, setTickerB] = useState("CVX");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("");
  const [significance, setSignificance] = useState(0.05);
  const [pairResult, setPairResult] = useState<PairAnalysisResponse | null>(null);
  const [pairLoading, setPairLoading] = useState(false);
  const [pairError, setPairError] = useState("");

  const analyzePair = async () => {
    setPairLoading(true);
    setPairError("");
    try {
      const res = await api.analyzePair({ ticker_a: tickerA.toUpperCase(), ticker_b: tickerB.toUpperCase(), start: startDate || "2015-01-01", end: endDate || undefined, significance });
      setPairResult(res);
    } catch (err: unknown) {
      setPairError(err instanceof Error ? err.message : String(err));
    } finally { setPairLoading(false); }
  };

  const [pairsText, setPairsText] = useState("XOM,CVX\nKO,PEP\nNVDA,META\nAAPL,MSFT");
  const [rankResult, setRankResult] = useState<PairData[] | null>(null);
  const [rankLoading, setRankLoading] = useState(false);
  const [rankError, setRankError] = useState("");

  const runRanking = async () => {
    setRankLoading(true); setRankError("");
    const pairs = pairsText.split("\n").map(l => l.trim()).filter(l => l.includes(",")).map(l => l.split(",").map(s => s.trim().toUpperCase()));
    if (pairs.length === 0) { setRankError("Enter at least one pair"); setRankLoading(false); return; }
    try { const res = await api.rankPairs({ pairs, start: startDate || "2015-01-01", end: endDate || undefined, significance, top_n: 10 }); setRankResult(res.ranked_pairs); }
    catch (err: unknown) { setRankError(err instanceof Error ? err.message : String(err)); }
    finally { setRankLoading(false); }
  };

  const [heatmapTickers, setHeatmapTickers] = useState("XOM,CVX,KO,PEP,NVDA,AMD,AAPL,MSFT,GOOGL,META");
  const [heatmapResult, setHeatmapResult] = useState<Record<string, Record<string, number>> | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState("");

  const runHeatmap = async () => {
    setHeatmapLoading(true); setHeatmapError("");
    const tickers = heatmapTickers.split(",").map(t => t.trim().toUpperCase()).filter(Boolean);
    if (tickers.length < 2) { setHeatmapError("Enter at least 2 tickers"); setHeatmapLoading(false); return; }
    try { const res = await api.buildHeatmap({ tickers, start: startDate || "2015-01-01", end: endDate || undefined, significance }); setHeatmapResult(res.matrix); }
    catch (err: unknown) { setHeatmapError(err instanceof Error ? err.message : String(err)); }
    finally { setHeatmapLoading(false); }
  };

  const navBtnStyle = (s: Section): React.CSSProperties => ({ padding: "6px 16px", fontWeight: section === s ? 700 : 400, background: section === s ? colors.primary : colors.tabInactive, color: section === s ? "#fff" : colors.text, border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13 });

  const p = pairResult?.pair;

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <div onClick={() => setHelpOpen(!helpOpen)} style={{ cursor: "pointer", fontSize: 13, color: colors.textMuted, userSelect: "none", display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, transition: "transform 0.15s", transform: helpOpen ? "rotate(90deg)" : "none" }}>▶</span>
          How to read this page
        </div>
        {helpOpen && (
          <div style={{ marginTop: 8, padding: 14, background: colors.surface, borderRadius: 6, fontSize: 13, lineHeight: 1.6, color: colors.text, borderLeft: `3px solid ${colors.primary}` }}>
            <p style={{ margin: "0 0 8px 0" }}><strong>What this does:</strong> Finds stock pairs that move together and tells you when one has drifted apart — a potential trade opportunity when they snap back.</p>
            <p style={{ margin: "0 0 6px 0", color: "#4CAF50" }}><strong>✅ Look for (green = good):</strong></p>
            <ul style={{ margin: "0 0 8px 0", paddingLeft: 20 }}>
              <li>"COINTEGRATED" badge — the pair has a stable long-term relationship</li>
              <li>"mean_reverting" regime — the spread tends to snap back (ideal for pair trading)</li>
              <li>Sharpe ratio &gt; 1 — good risk-adjusted returns in the backtest</li>
              <li>WF Backtest shows positive return — the strategy made money on unseen data</li>
              <li>Structural breaks = 0 — the relationship hasn't broken down</li>
            </ul>
            <p style={{ margin: "0 0 6px 0", color: "#F44336" }}><strong>🚩 Avoid (red flags):</strong></p>
            <ul style={{ margin: "0 0 8px 0", paddingLeft: 20 }}>
              <li>"NOT COINTEGRATED" — the stocks don't move together, don't trade this pair</li>
              <li>Structural break detected — the relationship may be permanently broken</li>
              <li>Regime = "trending" or "volatile" — mean-reversion strategies struggle here</li>
              <li>WF Backtest losing money — the strategy fails when tested honestly</li>
            </ul>
            <p style={{ margin: 0 }}><strong>💡 Tip:</strong> Score &gt; 0.5 on the Rank tab = a high-quality pair. The "Walk-Forward Backtest" is the honest result — trust it over the "In-Sample Backtest".</p>
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {(["analyze", "rank", "heatmap"] as Section[]).map(s => (
          <button key={s} style={navBtnStyle(s)} onClick={() => setSection(s)}>{s === "analyze" ? "Single Pair" : s === "rank" ? "Rank Pairs" : "Heatmap"}</button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16, padding: 12, background: colors.surface, borderRadius: 6 }}>
        <label style={{ fontSize: 12, color: colors.text }}>Start Date <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} style={inputStyle} /></label>
        <label style={{ fontSize: 12, color: colors.text }}>End Date <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} style={{ ...inputStyle, width: 90 }} /></label>
        <label style={{ fontSize: 12, color: colors.text }}>α <select value={significance} onChange={e => setSignificance(Number(e.target.value))} style={{ ...inputStyle, width: 80 }}>{[0.01, 0.05, 0.10].map(v => <option key={v} value={v}>{v}</option>)}</select></label>
      </div>

      {section === "analyze" && (
        <div>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
            <label style={{ fontSize: 12, color: colors.text }}>A <input value={tickerA} onChange={e => setTickerA(e.target.value.toUpperCase())} style={inputStyle} /></label>
            <label style={{ fontSize: 12, color: colors.text }}>B <input value={tickerB} onChange={e => setTickerB(e.target.value.toUpperCase())} style={inputStyle} /></label>
            <button style={btnStyle(pairLoading)} onClick={analyzePair} disabled={pairLoading}>{pairLoading ? "Analyzing…" : "Analyze Pair"}</button>
          </div>
          {pairError && <div style={{ padding: 12, background: colors.negative + "18", color: colors.negative, borderRadius: 6, marginBottom: 16, fontSize: 13 }}>{pairError}</div>}
          {pairLoading && <div style={{ textAlign: "center", color: colors.textMuted, fontSize: 13, padding: 16 }}>Computing cointegration, spread, walk-forward, regimes, backtests…</div>}

          {p && !pairLoading && (
            <div>
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
                <h3 style={{ margin: 0 }}>{p.ticker_a} / {p.ticker_b}</h3>
                <span style={{ padding: "2px 10px", borderRadius: 12, fontSize: 12, fontWeight: 600, background: p.cointegration.is_cointegrated ? colors.positive + "22" : colors.negative + "22", color: p.cointegration.is_cointegrated ? colors.positive : colors.negative }}>
                  {p.cointegration.is_cointegrated ? "COINTEGRATED" : "NOT COINTEGRATED"}
                </span>
                <RegimeBadge regime={p.regime.current_regime} />
                {(p.regime.num_structural_breaks ?? 0) > 0 && <span style={{ fontSize: 11, color: "#FF9800" }}>{p.regime.num_structural_breaks} structural break{(p.regime.num_structural_breaks ?? 0) > 1 ? "s" : ""}</span>}
                {p.regime.trading_allowed ? null : <span style={{ fontSize: 11, color: colors.textMuted }}>trading disabled</span>}
                {p.regime.signal_suppressed && <span style={{ fontSize: 11, color: "#FF9800" }}>signals suppressed</span>}
              </div>

              {(() => {
                const isCoint = p.cointegration.is_cointegrated;
                const hasBreaks = (p.regime.num_structural_breaks ?? 0) > 0;
                const wfGood = (p.wf_backtest?.sharpe_ratio ?? 0) > 0.5;
                let label: string, color: string, bg: string, detail: string;
                if (!isCoint) {
                  label = "AVOID"; color = "#F44336"; bg = "#F4433618";
                  detail = "Not cointegrated — the stocks don't move together";
                } else if (hasBreaks) {
                  label = "AVOID"; color = "#F44336"; bg = "#F4433618";
                  detail = "Structural break detected — relationship may be broken";
                } else if (wfGood) {
                  label = "TRADE"; color = "#4CAF50"; bg = "#4CAF5018";
                  detail = "Cointegrated, no breaks, positive OOS backtest";
                } else {
                  label = "CAUTION"; color = "#FF9800"; bg = "#FF980018";
                  detail = "Cointegrated but OOS backtest is weak — proceed carefully";
                }
                return (
                  <div style={{ padding: "8px 14px", borderRadius: 6, background: bg, color, fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
                    {label} — {detail}
                  </div>
                );
              })()}

              {/* ── Cointegration ── */}
              <CollapsibleSection title="Cointegration">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                  <MetricCard label="EG p-value" value={pval(p.cointegration.p_value)} />
                  <MetricCard label="Test Stat" value={fmt(p.cointegration.test_statistic, 2)} />
                  <MetricCard label="Hedge Ratio" value={fmt(p.cointegration.hedge_ratio)} sub={p.cointegration.hedge_ratio_method} />
                  <MetricCard label="Intercept" value={fmt(p.cointegration.hedge_ratio_intercept)} />
                  {p.johansen && <MetricCard label="Johansen" value={p.johansen.is_cointegrated ? "YES" : "NO"} sub={`trace=${fmt(p.johansen.trace_statistic, 2)}`} color={p.johansen.is_cointegrated ? colors.positive : colors.negative} />}
                </div>
              </CollapsibleSection>

              {/* ── Spread ── */}
              <CollapsibleSection title="Spread">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                  <MetricCard label="Half-Life" value={fmt(p.spread.half_life, 1) + "d"} />
                  <MetricCard label="Hurst" value={fmt(p.spread.hurst_exponent, 3)} color={p.spread.hurst_exponent < 0.5 ? "#4CAF50" : "#FF9800"} />
                  <MetricCard label="Z-Score" value={fmt(p.spread.current_zscore, 2)} color={Math.abs(p.spread.current_zscore) > 2 ? "#4CAF50" : colors.text} />
                  <MetricCard label="Stationary" value={p.spread.is_stationary ? "YES" : "NO"} color={p.spread.is_stationary ? "#4CAF50" : "#F44336"} />
                  <MetricCard label="ADF p-value" value={pval(p.spread.adf_pvalue)} />
                  <MetricCard label="Speed θ" value={fmt(p.spread.mean_reversion_speed, 4)} />
                  <MetricCard label="Persistence" value={fmt(p.spread.persistence, 3)} />
                  <MetricCard label="Var Ratio" value={fmt(p.spread.variance_ratio, 3)} sub={(p.spread.variance_ratio ?? 1) > 1 ? "trending" : "reverting"} />
                  <MetricCard label="Autocorr 5d" value={fmt(p.spread.spread_autocorr_5, 3)} />
                  {(p.spread.expected_time_to_mean ?? 9999) < 9999 && <MetricCard label="Exp. Time-to-Mean" value={fmt(p.spread.expected_time_to_mean, 1) + "d"} />}
                  <MetricCard label="Var Explosion" value={fmt(p.spread.variance_explosion_events, 0)} sub="events" color={(p.spread.variance_explosion_events ?? 0) > 0 ? "#FF9800" : colors.text} />
                </div>
              </CollapsibleSection>

              {/* ── Regime & Structural Breaks ── */}
              <CollapsibleSection title="Regime & Structural Breaks">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                  <MetricCard label="Current" value={p.regime.current_regime.replace("_", " ")} sub={p.regime.trading_allowed ? "tradeable" : "not tradeable"} />
                  <MetricCard label="VIX" value={fmt(p.regime.vix_level, 1)} />
                  <MetricCard label="Corr Breakdown" value={p.regime.correlation_breakdown ? "YES" : "NO"} color={p.regime.correlation_breakdown ? "#F44336" : "#4CAF50"} />
                  <MetricCard label="Var Expansion" value={p.regime.spread_variance_expansion ? "YES" : "NO"} color={p.regime.spread_variance_expansion ? "#FF9800" : colors.text} />
                  <MetricCard label="Struct Break" value={p.regime.structural_break ? "YES" : "NO"} color={p.regime.structural_break ? "#F44336" : "#4CAF50"} />
                  <MetricCard label="CUSUM" value={p.regime.cusum_break_detected ? "BREAK" : "OK"} color={p.regime.cusum_break_detected ? "#FF9800" : "#4CAF50"} sub={`${(p.regime.cusum_break_indices ?? []).length} crossings`} />
                  <MetricCard label="Chow" value={p.regime.chow_break_detected ? "BREAK" : "OK"} color={p.regime.chow_break_detected ? "#FF9800" : "#4CAF50"} />
                  <MetricCard label="Bai-Perron" value={fmt((p.regime.bai_perron_breaks ?? []).length, 0)} sub="breaks" color={(p.regime.bai_perron_breaks ?? []).length > 0 ? "#FF9800" : "#4CAF50"} />
                  <MetricCard label="Total Breaks" value={fmt(p.regime.num_structural_breaks, 0)} color={(p.regime.num_structural_breaks ?? 0) > 0 ? "#F44336" : "#4CAF50"} />
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                  <MetricCard label="Mean Rev" value={fmt((p.regime.regime_summary?.mean_reverting || 0) * 100, 1) + "%"} />
                  <MetricCard label="Trending" value={fmt((p.regime.regime_summary?.trending || 0) * 100, 1) + "%"} />
                  <MetricCard label="Volatile" value={fmt((p.regime.regime_summary?.volatile || 0) * 100, 1) + "%"} />
                  <MetricCard label="Choppy" value={fmt((p.regime.regime_summary?.choppy || 0) * 100, 1) + "%"} />
                </div>
                {(p.regime.chow_break_dates ?? []).length > 0 && (
                  <div style={{ fontSize: 12, color: colors.textMuted, marginBottom: 8 }}>
                    Chow break dates: {(p.regime.chow_break_dates ?? []).join(", ")}
                  </div>
                )}
              </CollapsibleSection>

              {/* ── Rolling Correlation ── */}
              <CollapsibleSection title="Rolling Correlation" defaultOpen={false}>
                {p.correlations.length > 0 && (
                  <div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                      {p.correlations.map(c => (
                        <div key={c.window} style={{ background: colors.surface, borderRadius: 6, padding: 10, minWidth: 160 }}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: colors.text, marginBottom: 6 }}>{c.window}d Window</div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            <MetricCard label="Current" value={fmt(c.current, 3)} />
                            <MetricCard label="Mean" value={fmt(c.mean, 3)} />
                            <MetricCard label="Std" value={fmt(c.std, 3)} color={c.std < 0.15 ? "#4CAF50" : colors.text} />
                            <MetricCard label="Slope" value={fmt(c.overall_slope, 4)} color={c.overall_slope !== undefined ? colors.text : undefined} />
                            <MetricCard label="Max DD" value={fmt(c.max_correlation_drawdown, 3)} color={(c.max_correlation_drawdown ?? 0) > 0.3 ? "#FF9800" : colors.text} />
                            <MetricCard label="Stability" value={fmt(c.stability_score, 3)} color={(c.stability_score ?? 1) > 0.5 ? "#4CAF50" : colors.text} />
                            <MetricCard label="Collapse" value={c.correlation_collapse ? "YES" : "NO"} color={c.correlation_collapse ? "#F44336" : "#4CAF50"} />
                            {c.threshold_crossings && (
                              <MetricCard label="Crossings" value={fmt(c.threshold_crossings.below_0_5 ?? c.threshold_crossings["below_0.5"] ?? 0, 0)} sub={`<0.5: ${c.threshold_crossings.below_0_3 ?? c.threshold_crossings["below_0.3"] ?? 0}`} />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CollapsibleSection>

              {/* ── Walk-Forward ── */}
              <CollapsibleSection title="Walk-Forward Validation">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                  <MetricCard label="OOS P-Value" value={pval(p.walk_forward.avg_oos_p_value)} sub={`train: ${pval(p.walk_forward.avg_train_p_value)}`} color={(p.walk_forward.avg_oos_p_value ?? 1) < 0.05 ? "#4CAF50" : colors.text} />
                  <MetricCard label="Coint %" value={fmt(p.walk_forward.cointegration_percentage, 1) + "%"} color={p.walk_forward.cointegration_percentage > 50 ? colors.positive : colors.negative} />
                  <MetricCard label="Avg Half-Life" value={fmt(p.walk_forward.avg_half_life, 1) + "d"} />
                  <MetricCard label="Avg Sharpe" value={fmt(p.walk_forward.avg_spread_sharpe, 2)} color={p.walk_forward.avg_spread_sharpe > 0 ? colors.positive : colors.negative} />
                  <MetricCard label="β Stability" value={fmt(p.walk_forward.hedge_ratio_stability, 3)} />
                  <MetricCard label="Spread DD" value={fmt(p.walk_forward.avg_spread_drawdown, 1) + "%"} color={colors.negative} />
                  <MetricCard label="OOS Stationarity" value={fmt(p.walk_forward.oos_stationarity_pct, 1) + "%"} />
                  <MetricCard label="Regime Stability" value={fmt(p.walk_forward.regime_stability_pct, 1) + "%"} />
                  <MetricCard label="Folds" value={String(p.walk_forward.num_folds)} />
                </div>
              </CollapsibleSection>

              {/* ── In-Sample Backtest (Reference) ── */}
              <CollapsibleSection title="In-Sample Backtest (Reference Only)" defaultOpen={false}>
                <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 8, fontStyle: "italic" }}>
                  Full-sample backtest — contains lookahead bias. Use WF Backtest below for OOS results.
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                  <MetricCard label="Total Return" value={fmt(p.backtest.total_return_pct, 1) + "%"} color={p.backtest.total_return_pct >= 0 ? colors.positive : colors.negative} />
                  <MetricCard label="Ann. Return" value={fmt(p.backtest.annualised_return_pct, 1) + "%"} color={p.backtest.annualised_return_pct && p.backtest.annualised_return_pct >= 0 ? colors.positive : colors.negative} />
                  <MetricCard label="Sharpe" value={fmt(p.backtest.sharpe_ratio, 2)} color={p.backtest.sharpe_ratio >= 1 ? colors.positive : p.backtest.sharpe_ratio > 0 ? colors.text : colors.negative} />
                  <MetricCard label="Sortino" value={fmt(p.backtest.sortino_ratio, 2)} color={p.backtest.sortino_ratio && p.backtest.sortino_ratio >= 1 ? colors.positive : colors.text} />
                  <MetricCard label="Calmar" value={fmt(p.backtest.calmar_ratio, 2)} />
                  <MetricCard label="Max DD" value={fmt(Math.abs(p.backtest.max_drawdown_pct), 1) + "%"} color={colors.negative} />
                  <MetricCard label="Win Rate" value={fmt(p.backtest.win_rate_pct, 1) + "%"} />
                  <MetricCard label="Trades" value={String(p.backtest.num_trades)} />
                  <MetricCard label="Avg Hold" value={fmt(p.backtest.avg_holding_period, 1) + "d"} />
                  <MetricCard label="Profit Factor" value={fmt(p.backtest.profit_factor, 2)} color={p.backtest.profit_factor && p.backtest.profit_factor > 1.5 ? colors.positive : colors.text} />
                  <MetricCard label="Exposure" value={fmt(p.backtest.exposure_pct, 1) + "%"} />
                  <MetricCard label="Turnover" value={fmt(p.backtest.turnover, 3)} />
                  <MetricCard label="β to Market" value={fmt(p.backtest.beta_to_market, 3)} />
                  <MetricCard label="Alpha" value={fmt(p.backtest.alpha, 3)} color={p.backtest.alpha && p.backtest.alpha > 0 ? colors.positive : colors.text} />
                  <MetricCard label="Info Ratio" value={fmt(p.backtest.information_ratio, 3)} color={p.backtest.information_ratio && p.backtest.information_ratio > 0.5 ? colors.positive : colors.text} />
                </div>
              </CollapsibleSection>

              {/* ── Walk-Forward Backtest (OOS) ── */}
              {p.wf_backtest && (
                <CollapsibleSection title="Walk-Forward Backtest (OOS — No Lookahead)">
                  <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 8, fontStyle: "italic" }}>
                    Out-of-sample backtest — each fold uses hedge ratio estimated on training data only.
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                    <MetricCard label="Total Return" value={fmt(p.wf_backtest.total_return_pct, 1) + "%"} color={(p.wf_backtest.total_return_pct ?? 0) >= 0 ? colors.positive : colors.negative} />
                    <MetricCard label="Ann. Return" value={fmt(p.wf_backtest.annualised_return_pct, 1) + "%"} color={(p.wf_backtest.annualised_return_pct ?? 0) >= 0 ? colors.positive : colors.negative} />
                    <MetricCard label="Sharpe" value={fmt(p.wf_backtest.sharpe_ratio, 2)} color={(p.wf_backtest.sharpe_ratio ?? 0) >= 1 ? colors.positive : (p.wf_backtest.sharpe_ratio ?? 0) > 0 ? colors.text : colors.negative} />
                    <MetricCard label="Max DD" value={fmt(Math.abs(p.wf_backtest.max_drawdown_pct ?? 0), 1) + "%"} color={colors.negative} />
                    <MetricCard label="Win Rate" value={fmt(p.wf_backtest.win_rate_pct, 1) + "%"} />
                    <MetricCard label="Trades" value={String(p.wf_backtest.num_trades ?? 0)} sub={`${p.wf_backtest.num_folds ?? 0} folds`} />
                    <MetricCard label="Avg Hold" value={fmt(p.wf_backtest.avg_holding_period, 1) + "d"} />
                    <MetricCard label="Turnover" value={fmt(p.wf_backtest.turnover, 3)} />
                  </div>
                </CollapsibleSection>
              )}

              {/* ── Charts ── */}
              <CollapsibleSection title="Charts">
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                  <ChartContainer data={p.spread.spread_series} color={colors.primary} label="Spread" height={280} />
                  <ChartContainer data={p.spread.zscore_series} color="#AA00FF" label="Z-Score" height={280} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                  <ChartContainer data={p.backtest.equity_curve} color={colors.positive} label="In-Sample PnL (Reference)" zeroLine={false} height={280} />
                  {p.backtest.drawdown_series && p.backtest.drawdown_series.length > 0 && (
                    <ChartContainer data={p.backtest.drawdown_series} color="#F44336" label="Drawdown" height={280} />
                  )}
                </div>
              </CollapsibleSection>
            </div>
          )}
        </div>
      )}

      {section === "rank" && (
        <div>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap", marginBottom: 16 }}>
            <label style={{ fontSize: 12, color: colors.text }}>
              Pairs (one per line, TICKER_A,TICKER_B)
              <textarea value={pairsText} onChange={e => setPairsText(e.target.value)} rows={6} style={{ display: "block", marginTop: 4, padding: "6px 10px", fontSize: 13, fontFamily: "monospace", border: `1px solid ${colors.border}`, borderRadius: 4, background: colors.surface, color: colors.text, width: 260, resize: "vertical" }} />
            </label>
            <button style={btnStyle(rankLoading)} onClick={runRanking} disabled={rankLoading}>{rankLoading ? "Ranking…" : "Rank Pairs"}</button>
          </div>
          {rankError && <div style={{ padding: 12, background: colors.negative + "18", color: colors.negative, borderRadius: 6, marginBottom: 16, fontSize: 13 }}>{rankError}</div>}
          {rankLoading && <div style={{ textAlign: "center", color: colors.textMuted, fontSize: 13, padding: 16 }}>Analyzing and ranking pairs…</div>}
          {rankResult && rankResult.length > 0 && !rankLoading && (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, color: colors.text }}>
                <thead>
                  <tr style={{ background: colors.surface, borderBottom: `1px solid ${colors.tableBorder}` }}>
                    {["#", "Pair", "Score", "EG p", "Adj p", "Coint", "β", "HL", "Hurst", "WF%", "OOS P", "Ret%", "Sharpe", "DD%", "Win%", "Trades", "Regime", "Breaks"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "left", fontWeight: 600, fontSize: 12, whiteSpace: "nowrap", borderBottom: `1px solid ${colors.tableBorder}` }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rankResult.map((pair, i) => (
                    <tr key={`${pair.ticker_a}-${pair.ticker_b}`} style={{ borderBottom: `1px solid ${colors.tableBorder}` }}>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{i + 1}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, fontWeight: 600 }}>{pair.ticker_a}/{pair.ticker_b}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.score, 3)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{pval(pair.cointegration.p_value)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{pval(pair.cointegration.p_value)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, color: pair.cointegration.is_cointegrated ? colors.positive : colors.negative }}>{pair.cointegration.is_cointegrated ? "✓" : "✗"}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.cointegration.hedge_ratio, 2)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.spread.half_life, 1)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.spread.hurst_exponent, 2)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.walk_forward.cointegration_percentage, 1)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{pval(pair.walk_forward.avg_oos_p_value)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, color: pair.backtest.total_return_pct >= 0 ? colors.positive : colors.negative }}>{fmt(pair.backtest.total_return_pct, 1)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.backtest.sharpe_ratio, 2)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, color: colors.negative }}>{fmt(Math.abs(pair.backtest.max_drawdown_pct), 1)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{fmt(pair.backtest.win_rate_pct, 1)}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{pair.backtest.num_trades}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12 }}>{pair.regime?.current_regime?.replace("_", " ") || "—"}</td>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, color: (pair.regime?.num_structural_breaks ?? 0) > 0 ? "#FF9800" : colors.text }}>{pair.regime?.num_structural_breaks ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {section === "heatmap" && (
        <div>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
            <label style={{ fontSize: 12, color: colors.text }}>Tickers (comma-separated) <input value={heatmapTickers} onChange={e => setHeatmapTickers(e.target.value.toUpperCase())} style={{ ...inputStyle, width: 400 }} /></label>
            <button style={btnStyle(heatmapLoading)} onClick={runHeatmap} disabled={heatmapLoading}>{heatmapLoading ? "Building…" : "Build Heatmap"}</button>
          </div>
          {heatmapError && <div style={{ padding: 12, background: colors.negative + "18", color: colors.negative, borderRadius: 6, marginBottom: 16, fontSize: 13 }}>{heatmapError}</div>}
          {heatmapLoading && <div style={{ textAlign: "center", color: colors.textMuted, fontSize: 13, padding: 16 }}>Computing pairwise cointegration scores…</div>}
          {heatmapResult && !heatmapLoading && (
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", fontSize: 13, color: colors.text }}>
                <thead>
                  <tr style={{ background: colors.surface }}>
                    <th style={{ padding: "8px 10px", textAlign: "left", fontWeight: 600, fontSize: 12, minWidth: 70 }}>Ticker</th>
                    {Object.keys(heatmapResult).map(t => <th key={t} style={{ padding: "8px 10px", textAlign: "left", fontWeight: 600, fontSize: 12, minWidth: 60 }}>{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(heatmapResult).map(([row, cols]) => (
                    <tr key={row} style={{ borderBottom: `1px solid ${colors.tableBorder}` }}>
                      <td style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, fontWeight: 600 }}>{row}</td>
                      {Object.keys(heatmapResult).map(col => {
                        const val = cols[col] ?? 0;
                        const sigThreshold = -Math.log10(significance);
                        const sig = val > sigThreshold;
                        const intensity = Math.min((val - sigThreshold) / 2, 1);
                        const bg = sig ? `rgba(76, 175, 80, ${0.2 + intensity * 0.4})` : "transparent";
                        return <td key={`${row}-${col}`} style={{ padding: "6px 10px", whiteSpace: "nowrap", fontSize: 12, background: bg, textAlign: "center", fontWeight: sig ? 700 : 400, color: sig ? "#1B5E20" : colors.text }}>{fmt(val, 2)}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 11, color: colors.textMuted, marginTop: 8 }}>
                Values = -log₁₀(p-value). <strong style={{ color: "#1B5E20" }}>Green (&gt; {-Math.log10(significance).toFixed(1)})</strong> = significant at p &lt; {significance}. Uncolored cells = not significant — the heatmap shows all pairs including weak ones for comparison.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
