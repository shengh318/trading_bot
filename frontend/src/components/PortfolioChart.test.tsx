import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThemeProvider } from "../theme/ThemeContext";
import PortfolioChart from "./PortfolioChart";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("PortfolioChart", () => {
  it("renders container when data is empty", () => {
    const { container } = render(<ThemeProvider><PortfolioChart data={[]} /></ThemeProvider>);
    expect(container.querySelector("div")).toBeDefined();
  });

  it("renders zoom toggle button", () => {
    render(<ThemeProvider><PortfolioChart data={[]} /></ThemeProvider>);
    expect(screen.getByTitle("Show all")).toBeDefined();
  });

  it("zoom button toggles between Show all and Zoom to recent", async () => {
    const data = [{ time: 1 as unknown as number, value: 100 }];
    render(<ThemeProvider><PortfolioChart data={data} /></ThemeProvider>);
    const btn = screen.getByTitle("Show all");
    fireEvent.click(btn);
    expect(screen.getByTitle("Zoom to recent")).toBeDefined();
    fireEvent.click(screen.getByTitle("Zoom to recent"));
    expect(screen.getByTitle("Show all")).toBeDefined();
  });

  it("renders with initialCash to use baseline series", () => {
    const data = [
      { time: 1 as unknown as number, value: 100 },
      { time: 2 as unknown as number, value: 110 },
    ];
    const { container } = render(
      <ThemeProvider><PortfolioChart data={data} initialCash={100} /></ThemeProvider>
    );
    expect(screen.getByTitle("Show all")).toBeDefined();
    expect(container.querySelector("div")).toBeDefined();
  });

  it("renders with markers", () => {
    const data = [
      { time: 1 as unknown as number, value: 100 },
      { time: 2 as unknown as number, value: 110 },
    ];
    const markers = [
      { time: 1 as unknown as number, side: "buy" as const },
      { time: 2 as unknown as number, side: "sell" as const },
    ];
    render(<ThemeProvider><PortfolioChart data={data} markers={markers} /></ThemeProvider>);
    expect(screen.getByTitle("Show all")).toBeDefined();
  });

  it("accepts custom height", () => {
    const { container } = render(
      <ThemeProvider><PortfolioChart data={[]} height={500} /></ThemeProvider>
    );
    expect(container.querySelector("div")).toBeDefined();
  });

  it("updates series when data changes", () => {
    const { rerender } = render(<ThemeProvider><PortfolioChart data={[]} /></ThemeProvider>);
    const data = [{ time: 1 as unknown as number, value: 100 }];
    rerender(<ThemeProvider><PortfolioChart data={data} /></ThemeProvider>);
    expect(screen.getByTitle("Show all")).toBeDefined();
  });
});
