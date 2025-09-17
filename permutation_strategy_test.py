#!/usr/bin/env python3
"""
Permutation Test – Multi-Metric, Multi-Asset (Strategy-aware)
============================================================

Now supports passing a Backtrader Strategy by name:

    python permutation_strategy_test.py \
        --symbols AAPL MSFT \
        --start 2015-01-01 --end 2024-12-31 \
        --n 20000 \
        --strategy SmaCross \
        --data-dirs DataManagement/data/alpha DataManagement/data/stooq \
        --combine fisher

If --strategy is omitted, falls back to the built-in SMA(50/200) signal
(identical to the original version).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yfinance as yf
from scipy.stats import combine_pvalues
from tqdm import tqdm

# ── NEW: backtrader imports for strategy signals ────────────────────────────
import backtrader as bt
import importlib

TRADING_DAYS = 252
METRICS = ["sharpe", "sortino", "max_dd"]

SUMMARY_COLS = [
    "symbol",
    "metric",
    "obs_value",
    "p_value",
    "holm_adj",
    "bh_adj",
]

# --------------------------- KPI helpers -----------------------------------

def annualised_ret_std(returns: pd.Series) -> Tuple[float, float]:
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    return mu, sigma

def compute_kpis(returns: pd.Series | pd.DataFrame) -> Dict[str, float]:
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("compute_kpis expects a single return series")
        returns = returns.squeeze("columns")

    mu, sigma = annualised_ret_std(returns)
    sharpe = mu / sigma if sigma != 0 else 0.0

    downside = returns[returns < 0]
    dd_sigma = downside.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = mu / dd_sigma if dd_sigma != 0 else 0.0

    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_dd = drawdown.min()

    return {"sharpe": sharpe, "sortino": sortino, "max_dd": max_dd}

# -------------------- Multiple-testing corrections -------------------------

def holm_bonferroni(p: List[float]) -> List[float]:
    m = len(p)
    idx_sorted = np.argsort(p)
    p_sorted = np.array(p)[idx_sorted]
    adj = np.empty(m)
    for k, pi in enumerate(p_sorted):
        adj[k] = min(1.0, (m - k) * pi)
    for k in range(1, m):
        adj[k] = max(adj[k], adj[k - 1])
    adj_unsorted = np.empty(m)
    adj_unsorted[idx_sorted] = adj
    return adj_unsorted.tolist()

def benjamini_hochberg(p: List[float]) -> List[float]:
    m = len(p)
    idx_sorted = np.argsort(p)
    p_sorted = np.array(p)[idx_sorted]
    adj = np.empty(m)
    prev = 1.0
    for k in reversed(range(m)):
        rank = k + 1
        adj[k] = min(prev, p_sorted[k] * m / rank)
        prev = adj[k]
    adj_unsorted = np.empty(m)
    adj_unsorted[idx_sorted] = adj
    return adj_unsorted.tolist()

# ---------------------------- Data helpers ---------------------------------

def fetch_price(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}.")
    return df

# ----------------------- Legacy (built-in) signals -------------------------

def builtin_sma_signals(price_df: pd.DataFrame) -> pd.Series:
    """Original SMA(50/200) rule (fallback)."""
    close = price_df["Adj Close"]
    sma_short = close.rolling(50).mean()
    sma_long = close.rolling(200).mean()
    return (sma_short > sma_long).astype(int).fillna(0)

# -------------------- NEW: Backtrader strategy signals ---------------------

class PositionSignal(bt.Analyzer):
    """Collect +1/0/−1 position direction per bar (same pattern as export_signals.py)."""
    def __init__(self):
        self._records: list[tuple[pd.Timestamp, int]] = []
    def next(self):
        dts = self.strategy.datetime.datetime(0)
        pos_dir = 0
        if self.strategy.position.size > 0:
            pos_dir = 1
        elif self.strategy.position.size < 0:
            pos_dir = -1
        self._records.append((pd.Timestamp(dts), pos_dir))
    def get_analysis(self):
        return pd.Series({dt: sig for dt, sig in self._records})

def locate_csv(symbol: str, roots: Sequence[str | Path]) -> Optional[Path]:
    for root in [Path(r) for r in roots]:
        cand = next(root.glob(f"{symbol}*.csv"), None)
        if cand:
            return cand
    return None

def strategy_signals_backtrader(
    symbol: str,
    start: str,
    end: str,
    strat_cls: type[bt.Strategy],
    data_dirs: Sequence[str | Path],
    capital: float = 10_000.0,
    commission: float = 0.0,
) -> pd.Series:
    """Run a minimal Backtrader session to extract +1/0/−1 signals from a Strategy."""
    data_path = locate_csv(symbol, data_dirs)
    if data_path is None:
        raise FileNotFoundError(
            f"Could not find CSV for {symbol} in: {', '.join(map(str, data_dirs))}"
        )

    tf = bt.TimeFrame.Minutes if "_m" in data_path.stem else bt.TimeFrame.Days
    comp = int(data_path.stem.split("_")[-1][:-1]) if "_m" in data_path.stem else 1
    fmt = "%Y-%m-%d %H:%M:%S" if tf is bt.TimeFrame.Minutes else "%Y-%m-%d"

    data = bt.feeds.GenericCSVData(
        dataname=str(data_path),
        dtformat=fmt,
        timeframe=tf,
        compression=comp,
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1,
        fromdate=pd.to_datetime(start),
        todate=pd.to_datetime(end),
    )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.adddata(data)
    cerebro.addstrategy(strat_cls)
    cerebro.addanalyzer(PositionSignal, _name="possig")

    res = cerebro.run()
    strat = res[0]
    ser: pd.Series = strat.analyzers.possig.get_analysis()
    ser.name = f"{strat_cls.__name__}_signal"
    return ser.sort_index()

# --------------------------- Returns combiner -------------------------------

def strategy_returns(returns: pd.Series, signals: pd.Series) -> pd.Series:
    """Position-weighted returns with 1-bar delay."""
    return signals.shift(1).reindex_like(returns).fillna(0) * returns

# ------------------------ Permutation core ---------------------------------

def permutation_test_metrics(
    observed_metrics: Dict[str, float],
    returns: pd.Series,
    signals: pd.Series,
    n_perm: int,
    seed: int | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    perms = {m: np.empty(n_perm) for m in METRICS}

    for _ in tqdm(range(n_perm), desc="Permuting", unit="perm", leave=False):
        shuff = signals.sample(frac=1.0, random_state=rng.integers(0, 2**32))
        shuff.index = signals.index
        tmp_metrics = compute_kpis(strategy_returns(returns, shuff))
        for m in METRICS:
            perms[m][_] = tmp_metrics[m]

    pvals = {}
    for m in METRICS:
        obs = observed_metrics[m]
        extreme = np.sum(np.abs(perms[m]) >= abs(obs))
        pvals[m] = (extreme + 1) / (n_perm + 1)
    return pvals, perms

# --------------------------- Per-symbol runner ------------------------------

def run_symbol(
    symbol: str,
    start: str,
    end: str,
    n: int,
    seed: int,
    out_root: Path,
    strategy_name: Optional[str],
    data_dirs: Sequence[str | Path],
) -> List[Dict]:
    price_df = fetch_price(symbol, start, end)
    daily_ret = price_df["Adj Close"].pct_change().dropna()

    # ---- choose signals: Backtrader strategy OR built-in SMA rule
    if strategy_name:
        # import your strats module and resolve the class
        strats_mod = importlib.import_module("strats")  # your file with Strategy classes
        lut = {cls.__name__: cls for cls in strats_mod.retall()}  # helper in your code
        if strategy_name not in lut:
            raise ValueError(f"Strategy '{strategy_name}' not found in strats.py")
        StratCls = lut[strategy_name]
        signals = strategy_signals_backtrader(
            symbol=symbol, start=start, end=end,
            strat_cls=StratCls, data_dirs=data_dirs
        )
        # align to daily return index
        signals = signals.reindex(daily_ret.index, method="pad").fillna(0).astype(int)
    else:
        signals = builtin_sma_signals(price_df)

    strat_ret = strategy_returns(daily_ret, signals)

    obs_metrics = compute_kpis(strat_ret)
    pvals, perms = permutation_test_metrics(obs_metrics, daily_ret, signals, n, seed)

    # --- save artefacts
    sym_dir = out_root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perms).to_csv(sym_dir / "permutation_metrics.csv", index=False)
    for m in METRICS:
        plt.figure(figsize=(9, 5))
        plt.hist(perms[m], bins=50, alpha=0.8, edgecolor="black")
        plt.axvline(obs_metrics[m], ls="--", lw=2, label="Observed")
        plt.title(f"Null distribution – {symbol} – {m} – {strategy_name or 'SMA50/200'}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(sym_dir / f"null_{m}.png")
        plt.close()

    raw = [pvals[m] for m in METRICS]
    holm = holm_bonferroni(raw)
    bh = benjamini_hochberg(raw)

    summary_rows = []
    for i, m in enumerate(METRICS):
        summary_rows.append(
            {
                "symbol":    symbol,
                "metric":    m,
                "obs_value": obs_metrics[m],
                "p_value":   raw[i],
                "holm_adj":  holm[i],
                "bh_adj":    bh[i],
            }
        )

    return summary_rows

# --------------------------------- CLI -------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Permutation test with optional Backtrader Strategy signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--combine", choices=["fisher", "stouffer"], default="fisher")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--seed", type=int, default=42)

    # NEW: choose a strategy and data locations
    parser.add_argument("--strategy", help="Backtrader Strategy class name from strats.py (e.g., SmaCross, Rsi2)")
    parser.add_argument("--data-dirs", nargs="*", default=[
        "DataManagement/data/alpha",  # same as your export_signals.py default
        "DataManagement/data/stooq",
        "data",
    ])

    args = parser.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    pvals_by_metric = {m: [] for m in METRICS}

    print("==============================")
    print(f"Strategy: {args.strategy or 'SMA50/200 (built-in)'}")
    for sym in args.symbols:
        print(f"Processing {sym} …")
        rows = run_symbol(
            sym, args.start, args.end, args.n, args.seed, out_root,
            strategy_name=args.strategy,
            data_dirs=args.data_dirs,
        )
        all_rows.extend(rows)
        for r in rows:
            pvals_by_metric[r["metric"]].append(r["p_value"])

    for m in METRICS:
        combined = combine_pvalues(pvals_by_metric[m], method=args.combine)[1]
        all_rows.append({
            "symbol":    "COMBINED",
            "metric":    m,
            "obs_value": "",
            "p_value":   combined,
            "holm_adj":  "",
            "bh_adj":    "",
        })

    pd.DataFrame(all_rows)[SUMMARY_COLS].to_csv(out_root / "summary_metrics.csv", index=False)
    print("==============================")
    print("Combined p-values across symbols (method: %s)" % args.combine)
    for m in METRICS:
        print(f"  {m:8s}: {combine_pvalues(pvals_by_metric[m], method=args.combine)[1]:.6f}")
    print("Results saved → %s" % (out_root / "summary_metrics.csv"))

if __name__ == "__main__":
    main()
