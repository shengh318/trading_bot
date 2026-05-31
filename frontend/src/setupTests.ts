import { vi } from "vitest";

vi.mock("lightweight-charts", () => {
  const mockSeries = {
    setData: vi.fn(),
    applyOptions: vi.fn(),
  };
  const mockChart = {
    addAreaSeries: vi.fn(() => mockSeries),
    addLineSeries: vi.fn(() => mockSeries),
    addBaselineSeries: vi.fn(() => mockSeries),
    applyOptions: vi.fn(),
    remove: vi.fn(),
    timeScale: vi.fn(() => ({
      fitContent: vi.fn(),
      applyOptions: vi.fn(),
      setVisibleLogicalRange: vi.fn(),
    })),
  };
  return {
    createChart: vi.fn(() => mockChart),
    ColorType: { Solid: "solid" },
    isBusinessDay: vi.fn(),
    isUTCTimestamp: vi.fn(),
  };
});

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});
