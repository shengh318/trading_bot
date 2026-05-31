import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { ThemeProvider } from "../theme/ThemeContext";
import Clock from "./Clock";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2024-06-15T14:30:25"));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Clock", () => {
  function renderClock() {
    return render(<ThemeProvider><Clock /></ThemeProvider>);
  }

  it("renders time in HH:MM:SS AM/PM format", () => {
    renderClock();
    expect(screen.getByText("02:30:25 PM")).toBeDefined();
  });

  it("renders date in readable format", () => {
    renderClock();
    expect(screen.getByText(/Jun 15, 2024/)).toBeDefined();
    expect(screen.getByText(/Sat/)).toBeDefined();
  });

  it("updates time every second", () => {
    renderClock();
    expect(screen.getByText("02:30:25 PM")).toBeDefined();

    act(() => { vi.advanceTimersByTime(1000); });
    expect(screen.getByText("02:30:26 PM")).toBeDefined();

    act(() => { vi.advanceTimersByTime(3000); });
    expect(screen.getByText("02:30:29 PM")).toBeDefined();
  });

  it("uses monospace font for time display", () => {
    renderClock();
    const timeDiv = screen.getByText("02:30:25 PM");
    expect(timeDiv.style.fontFamily).toBe("monospace");
  });

  it("cleans up interval on unmount", () => {
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
    const { unmount } = renderClock();
    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
  });

  it("shows midnight as 12:00:00 AM", () => {
    vi.setSystemTime(new Date("2024-06-15T00:00:00"));
    renderClock();
    expect(screen.getByText("12:00:00 AM")).toBeDefined();
  });

  it("shows noon as 12:00:00 PM", () => {
    vi.setSystemTime(new Date("2024-06-15T12:00:00"));
    renderClock();
    expect(screen.getByText("12:00:00 PM")).toBeDefined();
  });
});
