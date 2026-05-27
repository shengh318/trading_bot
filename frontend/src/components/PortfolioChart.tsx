import { useEffect, useRef } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type SeriesMarker, type AreaSeriesPartialOptions, type Time } from "lightweight-charts";

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
}

export default function PortfolioChart({ data, markers = [], height = 350 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      height,
      layout: { background: { type: ColorType.Solid, color: "white" } },
      grid: {
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      timeScale: { borderColor: "#ddd" },
      rightPriceScale: { borderColor: "#ddd" },
    });
    const series = chart.addAreaSeries({
      lineColor: "#1a73e8",
      topColor: "rgba(26,115,232,0.3)",
      bottomColor: "rgba(26,115,232,0.02)",
    } satisfies AreaSeriesPartialOptions);
    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      chart.applyOptions({ width: containerRef.current!.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    if (seriesRef.current && data.length > 0) {
      seriesRef.current.setData(data);
      const chartMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
        time: m.time,
        position: m.side === "buy" ? "belowBar" : "aboveBar",
        shape: m.side === "buy" ? "arrowUp" : "arrowDown",
        color: m.side === "buy" ? "#0b8043" : "#c5221f",
        size: 1,
      }));
      seriesRef.current.setMarkers(chartMarkers);
    }
  }, [data, markers]);

  return <div ref={containerRef} style={{ width: "100%", marginBottom: 16 }} />;
}
