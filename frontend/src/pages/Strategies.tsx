import { useEffect, useState } from "react";
import { api, type StrategyInfo } from "../api/client";

export default function Strategies() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);

  useEffect(() => {
    api.getStrategies().then(setStrategies);
  }, []);

  return (
    <div>
      <h3 style={{ margin: "0 0 12px" }}>Available Strategies</h3>
      {strategies.length === 0 ? (
        <p style={{ color: "#777" }}>No strategies found</p>
      ) : (
        strategies.map((s) => (
          <div
            key={s.name}
            style={{
              padding: 16,
              marginBottom: 12,
              border: "1px solid #e0e0e0",
              borderRadius: 8,
              background: "#f8f9fa",
            }}
          >
            <h4 style={{ margin: "0 0 4px" }}>{s.name}</h4>
            <p style={{ margin: "0 0 8px", fontSize: 13, color: "#555" }}>
              {s.description}
            </p>
            {s.params.length > 0 && (
              <div>
                <strong style={{ fontSize: 13 }}>Parameters:</strong>
                <table style={{ marginTop: 4, fontSize: 13, borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: "4px 12px 4px 0", borderBottom: "1px solid #ddd" }}>Name</th>
                      <th style={{ textAlign: "left", padding: "4px 12px", borderBottom: "1px solid #ddd" }}>Type</th>
                      <th style={{ textAlign: "left", padding: "4px 0 4px 12px", borderBottom: "1px solid #ddd" }}>Default</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.params.map((p) => (
                      <tr key={p.name}>
                        <td style={{ padding: "4px 12px 4px 0" }}>{p.name}</td>
                        <td style={{ padding: "4px 12px" }}>{p.type}</td>
                        <td style={{ padding: "4px 0 4px 12px" }}>
                          {String(p.default)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}
