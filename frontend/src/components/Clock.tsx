import { useEffect, useState } from "react";
import { useTheme } from "../theme/ThemeContext";

export default function Clock() {
  const { colors } = useTheme();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const pad = (n: number) => n.toString().padStart(2, "0");
  const h = now.getHours();
  const hh = pad(h % 12 || 12);
  const mm = pad(now.getMinutes());
  const ss = pad(now.getSeconds());
  const ampm = h < 12 ? "AM" : "PM";
  const date = now.toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  return (
    <div style={{ textAlign: "center", lineHeight: 1.3 }}>
      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace", color: colors.text }}>
        {hh}:{mm}:{ss} {ampm}
      </div>
      <div style={{ fontSize: 12, color: colors.textMuted }}>{date}</div>
    </div>
  );
}
