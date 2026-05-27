import { useEffect, useRef, useState, useCallback } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type SeriesMarker, type Time } from "lightweight-charts";
import { useTheme } from "../theme/ThemeContext";

interface Point {
  time: Time;
  value: number;
}

interface Marker {
  time: Time;
  side: "buy" | "sell";
}

interface Props {
  data: Point[];
  markers?: Marker[];
  height?: number;
  initialCash?: number;
}

export default function PortfolioChart({ data, markers = [], height = 350, initialCash }: Props) {
  const { colors } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | ISeriesApi<"Baseline"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      height,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text },
      grid: {
        vertLines: { color: colors.chart.gridColor },
        horzLines: { color: colors.chart.gridColor },
      },
      timeScale: { borderColor: colors.chart.borderColor },
      rightPriceScale: { borderColor: colors.chart.borderColor },
    });

    let series: ISeriesApi<"Area"> | ISeriesApi<"Baseline">;
    if (initialCash != null) {
      series = chart.addBaselineSeries({
        baseValue: { type: "price", price: initialCash },
        topLineColor: colors.positive,
        topFillColor1: colors.positive + "40",
        topFillColor2: colors.positive + "05",
        bottomLineColor: colors.negative,
        bottomFillColor1: colors.negative + "40",
        bottomFillColor2: colors.negative + "05",
        lineWidth: 2,
      });
    } else {
      series = chart.addAreaSeries({
        lineColor: colors.chart.lineColor,
        topColor: colors.chart.topColor,
        bottomColor: colors.chart.bottomColor,
      });
    }

    if (data.length > 0) {
      series.setData(data);
      const chartMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
        time: m.time,
        position: m.side === "buy" ? "belowBar" : "aboveBar",
        shape: m.side === "buy" ? "arrowUp" : "arrowDown",
        color: m.side === "buy" ? colors.positive : colors.negative,
        size: 1,
      }));
      series.setMarkers(chartMarkers);
    }

    seriesRef.current = series;
    chartRef.current = chart;

    const handleResize = () => {
      chart.applyOptions({ width: containerRef.current!.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [height, colors, initialCash]);

  useEffect(() => {
    if (!seriesRef.current) return;
    if (data.length === 0) {
      seriesRef.current.setData([]);
      seriesRef.current.setMarkers([]);
      return;
    }
    seriesRef.current.setData(data);
    const chartMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
      time: m.time,
      position: m.side === "buy" ? "belowBar" : "aboveBar",
      shape: m.side === "buy" ? "arrowUp" : "arrowDown",
      color: m.side === "buy" ? colors.positive : colors.negative,
      size: 1,
    }));
    seriesRef.current.setMarkers(chartMarkers);
  }, [data, markers, colors.positive, colors.negative]);

  const [fitMode, setFitMode] = useState(false);

  const toggleZoom = useCallback(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (fitMode) {
      const len = data.length;
      chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, len - 60), to: len });
      setFitMode(false);
    } else {
      chart.timeScale().fitContent();
      setFitMode(true);
    }
  }, [fitMode, data.length]);

  return (
    <div style={{ position: "relative", width: "100%", marginBottom: 16 }}>
      <button
        onClick={toggleZoom}
        title={fitMode ? "Zoom to recent" : "Show all"}
        style={{
          position: "absolute",
          top: 8,
          left: 8,
          zIndex: 10,
          width: 28,
          height: 28,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: colors.surface,
          color: colors.text,
          border: `1px solid ${colors.border}`,
          borderRadius: 4,
          cursor: "pointer",
          fontSize: 14,
          lineHeight: 1,
        }}
      >
        {fitMode ? "+" : "−"}
      </button>
      <div ref={containerRef} style={{ width: "100%" }} />
    </div>
  );
}
