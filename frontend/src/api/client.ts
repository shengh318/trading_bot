export interface AccountSummary {
  cash: number;
  portfolio_value: number;
  buying_power: number;
  day_pnl: number;
}

export interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  current_price: number;
  unrealized_pl: number;
  market_value: number;
}

export interface Order {
  id: string;
  symbol: string;
  side: string;
  qty: number;
  filled_qty: number;
  filled_avg_price: number | null;
  status: string;
  type: string;
  created_at: string;
  updated_at: string;
}

export interface StrategyParamInfo {
  name: string;
  type: string;
  default: unknown;
}

export interface StrategyInfo {
  name: string;
  description: string;
  params: StrategyParamInfo[];
}

export interface BacktestRunRequest {
  strategy_name: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_cash: number;
  parameters?: Record<string, unknown>;
}

export interface BacktestMetrics {
  total_return_pct: number;
  final_equity: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  win_rate_pct: number;
  num_trades: number;
  profit_factor: number;
}

export interface BacktestRun {
  id: number;
  strategy_name: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_cash: number;
  metrics: BacktestMetrics | null;
  created_at: string;
}

export interface EquityPoint {
  timestamp: string;
  total_equity: number;
  cash: number;
}

export interface Trade {
  bar_index: number;
  timestamp: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  pnl: number | null;
}

export interface Snapshot {
  bar_index: number;
  timestamp: string;
  equity: number;
  cash: number;
}

export interface WsBarEvent {
  type: "bar";
  bar_index: number;
  timestamp: string;
  equity: number;
  cash: number;
  signal: string;
  trade: Trade | null;
}

export interface WsCompleteEvent {
  type: "complete";
  run_id: number;
  metrics?: BacktestRun;
}

export interface WsErrorEvent {
  type: "error";
  message: string;
}

export type WsEvent = WsBarEvent | WsCompleteEvent | WsErrorEvent;

export class ApiClient {
  private base = "";

  async getStrategies(): Promise<StrategyInfo[]> {
    const res = await fetch(`${this.base}/api/strategies`);
    return res.json();
  }

  async getAccountSummary(): Promise<AccountSummary> {
    const res = await fetch(`${this.base}/api/portfolio/summary`);
    return res.json();
  }

  async getEquityCurve(limit = 500): Promise<EquityPoint[]> {
    const res = await fetch(`${this.base}/api/portfolio/equity-curve?limit=${limit}`);
    return res.json();
  }

  async getPositions(): Promise<Position[]> {
    const res = await fetch(`${this.base}/api/positions`);
    return res.json();
  }

  async getOrders(limit = 100): Promise<Order[]> {
    const res = await fetch(`${this.base}/api/orders?limit=${limit}`);
    return res.json();
  }

  async getBacktestRuns(limit = 20): Promise<BacktestRun[]> {
    const res = await fetch(`${this.base}/api/backtest/runs?limit=${limit}`);
    return res.json();
  }

  async getBacktestRun(id: number): Promise<BacktestRun> {
    const res = await fetch(`${this.base}/api/backtest/runs/${id}`);
    return res.json();
  }

  async getBacktestTrades(id: number): Promise<Trade[]> {
    const res = await fetch(`${this.base}/api/backtest/runs/${id}/trades`);
    return res.json();
  }

  async getBacktestEquity(id: number): Promise<Snapshot[]> {
    const res = await fetch(`${this.base}/api/backtest/runs/${id}/equity`);
    return res.json();
  }

  async runBacktest(req: BacktestRunRequest): Promise<BacktestRun> {
    const res = await fetch(`${this.base}/api/backtest/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    return res.json();
  }

  createBacktestSocket(): BacktestSocket {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws/backtest`;
    return new BacktestSocket(wsUrl);
  }
}

export type WsCallback = {
  onBar?: (event: WsBarEvent) => void;
  onComplete?: (event: WsCompleteEvent) => void;
  onError?: (message: string) => void;
};

export class BacktestSocket {
  private ws: WebSocket | null = null;
  private callbacks: WsCallback = {};

  constructor(private url: string) {}

  connect(callbacks: WsCallback) {
    this.callbacks = callbacks;
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = (msg) => {
      const event: WsEvent = JSON.parse(msg.data);
      switch (event.type) {
        case "bar":
          this.callbacks.onBar?.(event);
          break;
        case "complete":
          this.callbacks.onComplete?.(event);
          break;
        case "error":
          this.callbacks.onError?.(event.message);
          break;
      }
    };
    return new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve();
      this.ws!.onerror = () => reject(new Error("WebSocket connection failed"));
    });
  }

  send(data: Record<string, unknown>) {
    this.ws?.send(JSON.stringify(data));
  }

  close() {
    this.ws?.close();
    this.ws = null;
  }
}

export const api = new ApiClient();
