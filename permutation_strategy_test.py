#!/usr/bin/env python3
"""
Permutation Test – Multi‑Metric, Multi‑Asset
===========================================
Évalue la significativité statistique d’une même **règle de trading** sur une
liste de symboles ; pour **chaque actif** :

* Calcule trois KPI : **Sharpe**, **Sortino**, **Max Drawdown**.
* Obtient des p-values par **test de permutation** (nul : signaux aléatoires).
* Applique deux corrections de multiplicité : **Holm‑Bonferroni** (FWER) et
  **Benjamini‑Hochberg** (FDR).

Enfin, agrège les p-values par *métrique* sur l’ensemble des symboles (méthodes
Fisher ou Stouffer).

Usage :
```
python permutation_strategy_test.py \
    --symbols AAPL MSFT GOOG \
    --start 2000-01-01 \
    --end   2020-01-01 \
    --n 50000              # permutations par actif
    --combine fisher       # ou stouffer
```
Chaque appel crée :
* dossier `output/<TICKER>/` : CSV permutation + histogrammes pour **chaque KPI**.
* fichier `output/summary_metrics.csv` : p‑values brutes + ajustées par symbole
  et par KPI ainsi que les p‑values combinées (ligne `COMBINED`).

Dépendances : `pandas`, `numpy`, `yfinance`, `matplotlib`, `tqdm`, `scipy`
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interactif – évite les soucis Tk
import matplotlib.pyplot as plt  # noqa: E402
import yfinance as yf
from scipy.stats import combine_pvalues
from tqdm import tqdm

TRADING_DAYS = 252  # annualisation factor
METRICS = ["sharpe", "sortino", "max_dd"]

# ---------------------------------------------------------------------------
# Helpers – stats & KPI
# ---------------------------------------------------------------------------

def annualised_ret_std(returns: pd.Series) -> Tuple[float, float]:
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    return mu, sigma

def compute_kpis(returns: pd.Series | pd.DataFrame) -> Dict[str, float]:
    """Compute Sharpe, Sortino and Max Drawdown for a *single* return series.

    If a DataFrame with a single column is passed, it is squeezed to a Series.
    """
    # Ensure 1‑D Series
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("compute_kpis expects a single return series, not a DataFrame with multiple columns.")
        returns = returns.squeeze("columns")

    mu, sigma = annualised_ret_std(returns)
    sharpe = mu / sigma if sigma != 0 else 0.0

    downside = returns[returns < 0]
    dd_sigma = downside.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = mu / dd_sigma if dd_sigma != 0 else 0.0

    # Max drawdown (most negative peak‑to‑trough). Keep sign (<= 0)
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_dd = drawdown.min()

    return {"sharpe": sharpe, "sortino": sortino, "max_dd": max_dd}


# ---------------------------------------------------------------------------
# Multiple‑testing corrections
# ---------------------------------------------------------------------------

def holm_bonferroni(p: List[float]) -> List[float]:
    m = len(p)
    idx_sorted = np.argsort(p)
    p_sorted = np.array(p)[idx_sorted]
    adj = np.empty(m)
    for k, pi in enumerate(p_sorted):
        adj[k] = min(1.0, (m - k) * pi)
    # enforce monotonicity (step‑down)
    for k in range(1, m):
        adj[k] = max(adj[k], adj[k - 1])
    # return to original order
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


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def fetch_price(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}.")
    return df


def generate_signals(price_df: pd.DataFrame) -> pd.Series:
    close = price_df["Adj Close"]
    sma_short = close.rolling(50).mean()
    sma_long = close.rolling(200).mean()
    return (sma_short > sma_long).astype(int).fillna(0)


def strategy_returns(returns: pd.Series, signals: pd.Series) -> pd.Series:
    return signals.shift(1).reindex_like(returns).fillna(0) * returns


# ---------------------------------------------------------------------------
# Permutation test – returns **dict(metric → p‑value)**
# ---------------------------------------------------------------------------

def permutation_test_metrics(
    observed_metrics: Dict[str, float],
    returns: pd.Series,
    signals: pd.Series,
    n_perm: int,
    seed: int | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    perms = {m: np.empty(n_perm) for m in METRICS}

    for i in tqdm(range(n_perm), desc="Permuting", unit="perm", leave=False):
        shuff = signals.sample(frac=1.0, random_state=rng.integers(0, 2**32))
        shuff.index = signals.index
        tmp_metrics = compute_kpis(strategy_returns(returns, shuff))
        for m in METRICS:
            perms[m][i] = tmp_metrics[m]

    pvals = {}
    for m in METRICS:
        obs = observed_metrics[m]
        # two‑tailed: absolute value comparison
        extreme = np.sum(np.abs(perms[m]) >= abs(obs))
        pvals[m] = (extreme + 1) / (n_perm + 1)
    return pvals, perms


# ---------------------------------------------------------------------------
# Per‑symbol runner
# ---------------------------------------------------------------------------

def run_symbol(
    symbol: str,
    start: str,
    end: str,
    n: int,
    seed: int,
    out_root: Path,
) -> Dict[str, float]:
    price_df = fetch_price(symbol, start, end)
    daily_ret = price_df["Adj Close"].pct_change().dropna()
    signals = generate_signals(price_df)
    strat_ret = strategy_returns(daily_ret, signals)

    obs_metrics = compute_kpis(strat_ret)
    pvals, perms = permutation_test_metrics(obs_metrics, daily_ret, signals, n, seed)

    # --- save artefacts
    sym_dir = out_root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    # permutation CSV (N×3)
    pd.DataFrame(perms).to_csv(sym_dir / "permutation_metrics.csv", index=False)
    # histogram for chaque KPI
    for m in METRICS:
        plt.figure(figsize=(9, 5))
        plt.hist(perms[m], bins=50, alpha=0.8, edgecolor="black")
        plt.axvline(obs_metrics[m], ls="--", lw=2, label="Observed")
        plt.title(f"Null distribution – {symbol} – {m}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(sym_dir / f"null_{m}.png")
        plt.close()

    # multiplicity corrections (sur les 3 métriques)
    raw = [pvals[m] for m in METRICS]
    holm = holm_bonferroni(raw)
    bh = benjamini_hochberg(raw)

    summary_rows = []
    for i, m in enumerate(METRICS):
        summary_rows.append(
            {
                "symbol": symbol,
                "metric": m,
                "p_value": raw[i],
                "holm_adj": holm[i],
                "bh_adj": bh[i],
            }
        )

    return summary_rows


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Permutation test on multiple KPI with multiplicity correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--combine", choices=["fisher", "stouffer"], default="fisher")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    pvals_by_metric = {m: [] for m in METRICS}

    print("==============================")
    for sym in args.symbols:
        print(f"Processing {sym} …")
        rows = run_symbol(sym, args.start, args.end, args.n, args.seed, out_root)
        all_rows.extend(rows)
        for r in rows:
            pvals_by_metric[r["metric"]].append(r["p_value"])

    # Combined p‑values across symbols, per metric
    for m in METRICS:
        combined = combine_pvalues(pvals_by_metric[m], method=args.combine)[1]
        all_rows.append({
            "symbol": "COMBINED",
            "metric": m,
            "p_value": combined,
            "holm_adj": "",
            "bh_adj": "",
        })

    pd.DataFrame(all_rows).to_csv(out_root / "summary_metrics.csv", index=False)

    print("==============================")
    print("Combined p‑values across symbols (method: %s)" % args.combine)
    for m in METRICS:
        print(f"  {m:8s}: {combine_pvalues(pvals_by_metric[m], method=args.combine)[1]:.6f}")
    print("Results saved → %s" % (out_root / "summary_metrics.csv"))


if __name__ == "__main__":
    main()
