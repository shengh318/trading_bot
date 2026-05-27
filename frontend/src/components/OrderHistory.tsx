import type { Order } from "../api/client";

interface Props {
  orders: Order[];
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

export default function OrderHistory({ orders }: Props) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h3 style={{ margin: "0 0 8px" }}>Order History</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={headerStyle}>ID</th>
            <th style={headerStyle}>Symbol</th>
            <th style={headerStyle}>Side</th>
            <th style={headerStyle}>Qty</th>
            <th style={headerStyle}>Filled</th>
            <th style={headerStyle}>Filled Price</th>
            <th style={headerStyle}>Status</th>
            <th style={headerStyle}>Type</th>
            <th style={headerStyle}>Created</th>
          </tr>
        </thead>
        <tbody>
          {orders.length === 0 ? (
            <tr>
              <td style={cellStyle} colSpan={9}>
                No orders
              </td>
            </tr>
          ) : (
            orders.map((o) => (
              <tr key={o.id}>
                <td style={cellStyle}>{o.id.slice(0, 8)}</td>
                <td style={cellStyle}>{o.symbol}</td>
                <td
                  style={{
                    ...cellStyle,
                    color: o.side === "buy" ? "#0b8043" : "#c5221f",
                  }}
                >
                  {o.side}
                </td>
                <td style={cellStyle}>{o.qty}</td>
                <td style={cellStyle}>{o.filled_qty}</td>
                <td style={cellStyle}>
                  {o.filled_avg_price != null
                    ? `$${o.filled_avg_price.toFixed(2)}`
                    : "-"}
                </td>
                <td style={cellStyle}>{o.status}</td>
                <td style={cellStyle}>{o.type}</td>
                <td style={cellStyle}>
                  {new Date(o.created_at).toLocaleString()}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
