"""Debug script: run full pipeline on a few pairs and print backtest metrics."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from backend.stats_arb.pipeline import PairAnalyzer

pairs = [("XOM","CVX"), ("JPM","GS"), ("KO","PEP"), ("AAPL","MSFT"), ("UNH","CVS")]
for a,b in pairs:
    try:
        r = PairAnalyzer(a, b, "2015-01-01", capital=100000).analyze()
        bt = r.wf_backtest or r.in_sample_backtest
        coint = r.cointegration
        print(f"{a}/{b}:")
        print(f"  coint_p={coint.p_value:.6f}  hedge_ratio={coint.hedge_ratio:.4f}")
        if bt:
            print(f"  sharpe={bt.sharpe_ratio:.3f}  ret={bt.total_return_pct:.1f}%  dd={bt.max_drawdown_pct:.1f}%  trades={bt.num_trades}  win%={bt.win_rate_pct:.1f}%")
        else:
            print(f"  NO BACKTEST (wf={r.wf_backtest is not None}, insample={r.in_sample_backtest is not None})")
    except Exception as e:
        print(f"{a}/{b}: FAILED - {e}")
    print()
