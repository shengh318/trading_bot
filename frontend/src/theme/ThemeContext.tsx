import { createContext, useContext, useState, useCallback, useMemo, useEffect, type ReactNode } from "react";

export interface ThemeColors {
  bg: string;
  surface: string;
  border: string;
  tableBorder: string;
  inputBg: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  primary: string;
  positive: string;
  negative: string;
  tabInactive: string;
  logBg: string;
  logText: string;
  chart: {
    lineColor: string;
    topColor: string;
    bottomColor: string;
    gridColor: string;
    borderColor: string;
  };
}

const light: ThemeColors = {
  bg: "#ffffff",
  surface: "#f8f9fa",
  border: "#e0e0e0",
  tableBorder: "#ddd",
  inputBg: "#ffffff",
  text: "#000000",
  textSecondary: "#555555",
  textMuted: "#777777",
  primary: "#1a73e8",
  positive: "#0b8043",
  negative: "#c5221f",
  tabInactive: "#e8eaed",
  logBg: "#1e1e1e",
  logText: "#d4d4d4",
  chart: {
    lineColor: "#1a73e8",
    topColor: "rgba(26,115,232,0.3)",
    bottomColor: "rgba(26,115,232,0.02)",
    gridColor: "#f0f0f0",
    borderColor: "#ddd",
  },
};

const dark: ThemeColors = {
  bg: "#121212",
  surface: "#1e1e1e",
  border: "#333333",
  tableBorder: "#444444",
  inputBg: "#2a2a2a",
  text: "#e0e0e0",
  textSecondary: "#a0a0a0",
  textMuted: "#888888",
  primary: "#4a9eff",
  positive: "#4caf50",
  negative: "#ef5350",
  tabInactive: "#333333",
  logBg: "#0d0d0d",
  logText: "#d4d4d4",
  chart: {
    lineColor: "#4a9eff",
    topColor: "rgba(74,158,255,0.3)",
    bottomColor: "rgba(74,158,255,0.02)",
    gridColor: "#333333",
    borderColor: "#444444",
  },
};

interface ThemeContextValue {
  isDark: boolean;
  toggleTheme: () => void;
  colors: ThemeColors;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  const toggleTheme = useCallback(() => setIsDark((p) => !p), []);

  const colors = useMemo(() => (isDark ? dark : light), [isDark]);

  useEffect(() => {
    document.body.style.background = colors.bg;
    document.body.style.margin = "0";
  }, [colors.bg]);

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme, colors }}>
      <div style={{ background: colors.bg, color: colors.text, minHeight: "100vh" }}>
        {children}
      </div>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
