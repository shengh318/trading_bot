import type { Position } from "../api/client";

interface Props {
  positions: Position[];
}

const cellStyle: React.CSSProperties = {
  padding: "6px 12px",
  textAlign: "left",
  borderBottom: "1px solid #e0e0e0",
  fontSize: 13,
};

const headerStyle: React.CSSProperties = {
  ...cellStyle,
  fontWeight: 600,
  background: "#f8f9fa",
  borderTop: "1px solid #e0e0e0",
};

export default function PositionsTable({ positions }: Props) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h3 style={{ margin: "0 0 8px" }}>Holdings</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={headerStyle}>Symbol</th>
            <th style={headerStyle}>Qty</th>
            <th style={headerStyle}>Avg Buy Price</th>
            <th style={headerStyle}>Current</th>
            <th style={headerStyle}>Profit/Loss</th>
            <th style={headerStyle}>Current Value</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 ? (
            <tr>
              <td style={cellStyle} colSpan={6}>
                No holdings
              </td>
            </tr>
          ) : (
            positions.map((p) => (
              <tr key={p.symbol}>
                <td style={cellStyle}>{p.symbol}</td>
                <td style={cellStyle}>{p.qty}</td>
                <td style={cellStyle}>{`$${p.avg_entry_price.toFixed(2)}`}</td>
                <td style={cellStyle}>{`$${p.current_price.toFixed(2)}`}</td>
                <td
                  style={{
                    ...cellStyle,
                    color: p.unrealized_pl >= 0 ? "#0b8043" : "#c5221f",
                  }}
                >
                  {`$${p.unrealized_pl.toFixed(2)}`}
                </td>
                <td style={cellStyle}>{`$${p.market_value.toFixed(2)}`}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
