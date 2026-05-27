import type { StrategyInfo } from "../api/client";

interface Props {
  strategies: StrategyInfo[];
  selected: string;
  params: Record<string, unknown>;
  onSelect: (name: string) => void;
  onParamChange: (name: string, value: unknown) => void;
}

export default function StrategySelector({
  strategies,
  selected,
  params,
  onSelect,
  onParamChange,
}: Props) {
  const current = strategies.find((s) => s.name === selected);

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ fontWeight: 600, marginRight: 8 }}>Strategy:</label>
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        style={{ padding: "4px 8px", fontSize: 14 }}
      >
        {strategies.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name}
          </option>
        ))}
      </select>
      {current && (
        <p style={{ fontSize: 13, color: "#555", margin: "4px 0 0" }}>
          {current.description}
        </p>
      )}
      {current?.params.map((p) => (
        <div key={p.name} style={{ marginTop: 8 }}>
          <label style={{ marginRight: 8, fontSize: 13 }}>
            {p.name} ({p.type}):
          </label>
          <input
            type={p.type === "int" ? "number" : "text"}
            defaultValue={(params[p.name] ?? p.default) as string | number}
            onChange={(e) =>
              onParamChange(
                p.name,
                p.type === "int" ? parseInt(e.target.value, 10) : e.target.value,
              )
            }
            style={{ padding: "4px 8px", fontSize: 14, width: 120 }}
          />
        </div>
      ))}
    </div>
  );
}
