#!/usr/bin/env python3
"""
backtrader_yaml_runner.py — run Backtrader strategies driven by a YAML file.

YAML schema (minimal example):

```yaml
start: "2005-01-01"   # ISO‑8601 strings *recommended* — quotes keep them strings
end:   "2020-01-01"   # Unquoted dates also work (YAML turns them into `date`)
capital: 10000
commission: 0.001
output_csv: results.csv

runs:
  - symbol: SPY
    strategy: GoldenCross

  - symbol: AAPL
    strategy: Rsi2
```

Dates can be **quoted strings** *or* bare ISO dates (which PyYAML converts to
`datetime.date`). The helper `parse_date()` below converts everything into a
`datetime.datetime`, so you won’t hit `TypeError: fromisoformat: argument must
be str` again.
"""

import argparse
import csv
import importlib
import sys
from datetime import date, datetime, time
from pathlib import Path

import backtrader as bt
import yaml

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(value, default_iso: str) -> datetime:
    """Return a `datetime` regardless of whether *value* is str/date/datetime/None."""
    if value is None:
        return datetime.fromisoformat(default_iso)

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):  # but *not* datetime (handled above)
        return datetime.combine(value, time.min)

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # Last‑ditch fallback — accept YYYY‑MM‑DD only
            return datetime.strptime(value, "%Y-%m-%d")

    raise TypeError(f"Unsupported date value: {value!r} ({type(value)})")


def load_strategy(name: str):
    """Import *name* from local `strats.py`.

    Abort cleanly if the strategy is missing so users see a friendly error.
    """
    mod = importlib.import_module("strats")
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise SystemExit(f"Strategy '{name}' not found in strats.py") from exc


# ---------------------------------------------------------------------------
# Core runner (one symbol + strategy per call)
# ---------------------------------------------------------------------------

def run_single(symbol: str, strat_cls, cfg: dict, start: datetime, end: datetime):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cfg.get("capital", 10_000))
    cerebro.broker.setcommission(commission=cfg.get("commission", 0.001))
    cerebro.addstrategy(strat_cls)

    csv_path = HERE / f"{symbol}.csv"
    if not csv_path.exists():
        raise SystemExit(f"CSV data for {symbol} not found at {csv_path}")

    data = bt.feeds.GenericCSVData(
        dataname=str(csv_path),
        dtformat="%Y-%m-%d",
        timeframe=bt.TimeFrame.Days,
        fromdate=start,
        todate=end,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
    )
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    results = cerebro.run()[0]
    return {
        "symbol": symbol,
        "strategy": strat_cls.__name__,
        "final_value": cerebro.broker.getvalue(),
        "sharpe": (results.analyzers.sharpe.get_analysis().get("sharperatio")),
        "max_drawdown_pct": results.analyzers.dd.get_analysis().max.drawdown,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    start = parse_date(cfg.get("start"), "2005-01-01")
    end = parse_date(cfg.get("end"), "2020-01-01")

    results = []
    for run in cfg.get("runs", []):
        symbol = run["symbol"]
        strat_cls = load_strategy(run["strategy"])

        res = run_single(symbol, strat_cls, cfg, start, end)
        results.append(res)
        print(f"{symbol}-{strat_cls.__name__}: final value = {res['final_value']:.2f}")

    if not results:
        sys.exit("No runs defined in YAML – nothing to do.")

    csv_name = cfg.get("output_csv", "results.csv")
    with open(csv_name, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved CSV results to {csv_name} (\u2714)")


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Backtrader tests from a YAML file")
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to YAML config (default: config.yaml)"
    )
    args = parser.parse_args()

    cfg_file = Path(args.config)
    if not cfg_file.exists():
        sys.exit(f"Config file '{cfg_file}' not found.")

    main(cfg_file)
