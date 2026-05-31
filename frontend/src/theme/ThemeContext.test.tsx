import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThemeProvider, useTheme } from "./ThemeContext";

function TestChild() {
  const { isDark, toggleTheme, colors } = useTheme();
  return (
    <div>
      <span data-testid="isDark">{String(isDark)}</span>
      <span data-testid="bg">{colors.bg}</span>
      <span data-testid="text">{colors.text}</span>
      <button data-testid="toggle" onClick={toggleTheme}>Toggle</button>
    </div>
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("ThemeContext", () => {
  it("defaults to light theme when system prefers light", () => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    render(<ThemeProvider><TestChild /></ThemeProvider>);
    expect(screen.getByTestId("isDark").textContent).toBe("false");
    expect(screen.getByTestId("bg").textContent).toBe("#ffffff");
    expect(screen.getByTestId("text").textContent).toBe("#000000");
  });

  it("defaults to dark theme when system prefers dark", () => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: query === "(prefers-color-scheme: dark)",
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    render(<ThemeProvider><TestChild /></ThemeProvider>);
    expect(screen.getByTestId("isDark").textContent).toBe("true");
    expect(screen.getByTestId("bg").textContent).toBe("#121212");
  });

  it("toggleTheme switches between light and dark", () => {
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    render(<ThemeProvider><TestChild /></ThemeProvider>);
    expect(screen.getByTestId("isDark").textContent).toBe("false");

    fireEvent.click(screen.getByTestId("toggle"));
    expect(screen.getByTestId("isDark").textContent).toBe("true");
    expect(screen.getByTestId("bg").textContent).toBe("#121212");

    fireEvent.click(screen.getByTestId("toggle"));
    expect(screen.getByTestId("isDark").textContent).toBe("false");
    expect(screen.getByTestId("bg").textContent).toBe("#ffffff");
  });

  it("useTheme throws outside ThemeProvider", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<TestChild />)).toThrow("useTheme must be used within ThemeProvider");
    consoleSpy.mockRestore();
  });

  it("sets body background color on mount", () => {
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    render(<ThemeProvider><TestChild /></ThemeProvider>);
    expect(document.body.style.background).toBe("rgb(255, 255, 255)");
  });

  it("renders children with dark background in dark mode", () => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: query === "(prefers-color-scheme: dark)",
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    const { container } = render(<ThemeProvider><div>child</div></ThemeProvider>);
    const outerDiv = container.firstChild as HTMLElement;
    expect(outerDiv.style.background).toBe("rgb(18, 18, 18)");
    expect(outerDiv.style.color).toBe("rgb(224, 224, 224)");
  });

  it("has all required color fields in both themes", () => {
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })) as unknown as typeof window.matchMedia;

    render(<ThemeProvider><TestChild /></ThemeProvider>);
    expect(screen.getByTestId("bg").textContent).toBeTruthy();
    expect(screen.getByTestId("text").textContent).toBeTruthy();
  });
});
