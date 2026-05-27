import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StrategySelector from "./StrategySelector";

const strategies = [
  {
    name: "SmaCrossover",
    description: "Buy when short SMA crosses above long SMA.",
    params: [
      { name: "short_window", type: "int", default: 20 },
      { name: "long_window", type: "int", default: 50 },
    ],
  },
  {
    name: "BollingerBands",
    description: "Bollinger Bands strategy.",
    params: [{ name: "period", type: "int", default: 20 }],
  },
];

describe("StrategySelector", () => {
  it("renders strategy options", () => {
    render(
      <StrategySelector
        strategies={strategies}
        selected="SmaCrossover"
        params={{}}
        onSelect={vi.fn()}
        onParamChange={vi.fn()}
      />,
    );
    expect(screen.getByText("SmaCrossover")).toBeDefined();
    expect(screen.getByText("BollingerBands")).toBeDefined();
  });

  it("shows description for selected strategy", () => {
    render(
      <StrategySelector
        strategies={strategies}
        selected="SmaCrossover"
        params={{}}
        onSelect={vi.fn()}
        onParamChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Buy when short SMA crosses above long SMA.")).toBeDefined();
  });

  it("renders parameter inputs", () => {
    render(
      <StrategySelector
        strategies={strategies}
        selected="SmaCrossover"
        params={{ short_window: 20, long_window: 50 }}
        onSelect={vi.fn()}
        onParamChange={vi.fn()}
      />,
    );
    expect(screen.getByText("short_window (int):")).toBeDefined();
    expect(screen.getByText("long_window (int):")).toBeDefined();
  });

  it("calls onSelect when strategy changes", () => {
    const onSelect = vi.fn();
    render(
      <StrategySelector
        strategies={strategies}
        selected="SmaCrossover"
        params={{}}
        onSelect={onSelect}
        onParamChange={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "BollingerBands" } });
    expect(onSelect).toHaveBeenCalledWith("BollingerBands");
  });

  it("calls onParamChange when parameter input changes", () => {
    const onParamChange = vi.fn();
    render(
      <StrategySelector
        strategies={strategies}
        selected="SmaCrossover"
        params={{ short_window: 20, long_window: 50 }}
        onSelect={vi.fn()}
        onParamChange={onParamChange}
      />,
    );
    const inputs = screen.getAllByRole("spinbutton");
    fireEvent.change(inputs[0], { target: { value: "30" } });
    expect(onParamChange).toHaveBeenCalledWith("short_window", 30);
  });
});
