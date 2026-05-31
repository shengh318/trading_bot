import { memo, useEffect, useRef } from "react";
import { useTheme } from "../theme/ThemeContext";

function ClockInner() {
  const { colors } = useTheme();
  const timeRef = useRef<HTMLDivElement>(null);
  const dateRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const pad = (n: number) => n.toString().padStart(2, "0");
    const tick = () => {
      const now = new Date();
      const h = now.getHours();
      const hh = pad(h % 12 || 12);
      const mm = pad(now.getMinutes());
      const ss = pad(now.getSeconds());
      const ampm = h < 12 ? "AM" : "PM";
      if (timeRef.current) {
        timeRef.current.textContent = `${hh}:${mm}:${ss} ${ampm}`;
      }
      if (dateRef.current) {
        dateRef.current.textContent = now.toLocaleDateString(undefined, {
          weekday: "short",
          year: "numeric",
          month: "short",
          day: "numeric",
        });
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ textAlign: "center", lineHeight: 1.3 }}>
      <div ref={timeRef} style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace", color: colors.text }}>
        --:--:-- --
      </div>
      <div ref={dateRef} style={{ fontSize: 12, color: colors.textMuted }}>
        ---
      </div>
    </div>
  );
}

export default memo(ClockInner);

