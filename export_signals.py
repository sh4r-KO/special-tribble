#!/usr/bin/env python3
"""
export_signals.py ────────────────────────────────────────────────────────────
Generate **per‑ticker CSVs** with one column per **strategy position signal**.

Changes in this version
-----------------------
* **DATA_DIRS**: script now searches the same local folders you showed
  (`DataManagement/data/alpha` and `DataManagement/data/stooq`) before falling
  back to a plain `data/` folder. No more hard‑coded single path.
* Added `--data-dirs` CLI flag if you want to override / add search locations
  on the fly.

Everything else remains identical: run once and you’ll get
`signals/<TICKER>.csv` files ready for hypothesis testing.
"""
from __future__ import annotations

import argparse
import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import backtrader as bt
import pandas as pd
import yaml

# --------------------------------------------------------------------------- 
# Where to look for price CSVs
# --------------------------------------------------------------------------- 
DEFAULT_DATA_DIRS = [
    Path("DataManagement/data/alpha"),   # local Alpha‑Vantage / AV downloader
    Path("DataManagement/data/stooq"),   # Stooq daily files
    Path("data"),                        # generic fallback
]

# --------------------------------------------------------------------------- 
# Analyzer that stores the *position state* each bar (1, 0, −1)
# --------------------------------------------------------------------------- 
class PositionSignal(bt.Analyzer):
    """Collect the position direction (+1 long, −1 short, 0 flat) each bar."""

    def __init__(self):
        self._records: list[tuple[datetime, int]] = []

    def next(self):
        dts = self.strategy.datetime.datetime(0)  # bt → python datetime
        pos_dir = 0
        if self.strategy.position.size > 0:
            pos_dir = 1
        elif self.strategy.position.size < 0:
            pos_dir = -1
        self._records.append((dts, pos_dir))

    def get_analysis(self):
        return pd.Series({dt: sig for dt, sig in self._records})


# --------------------------------------------------------------------------- 
# YAML loader helper (symbols, dates, etc.)
# --------------------------------------------------------------------------- 
def load_config(path: str | os.PathLike) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "symbols" not in cfg or not cfg["symbols"]:
        raise ValueError("Config must define a non‑empty 'symbols' list")
    return cfg


# --------------------------------------------------------------------------- 
# Core export routine
# --------------------------------------------------------------------------- 

def export_signals(
    cfg: dict,
    outdir: str | os.PathLike = "signals",
    strats_to_use: Sequence[str] | None = None,
    commission: float | None = None,
    data_dirs: Sequence[str | os.PathLike] | None = None,
):
    """Run each selected strategy on every symbol and write per‑ticker CSVs."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Strategy classes from strats.py
    strats_mod = importlib.import_module("strats")
    all_strat_classes: List[type[bt.Strategy]] = strats_mod.retall()

    if strats_to_use is not None:
        all_strat_classes = [cls for cls in all_strat_classes if cls.__name__ in strats_to_use]
        missing = set(strats_to_use) - {cls.__name__ for cls in all_strat_classes}
        if missing:
            raise ValueError(f"Unknown strategies in --strats: {', '.join(sorted(missing))}")

    if not all_strat_classes:
        raise RuntimeError("No strategies selected – nothing to do.")

    # Prepare date bounds
    start = datetime.fromisoformat(cfg.get("start", "1900-01-01"))
    end = datetime.fromisoformat(cfg.get("end", datetime.today().strftime("%Y-%m-%d")))

    # Data folders
    search_dirs = [Path(p) for p in (data_dirs or DEFAULT_DATA_DIRS)]

    for symbol in cfg["symbols"]:
        print(f"→ {symbol}")
        sig_frames: list[pd.Series] = []

        for StratCls in all_strat_classes:
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(cfg.get("capital", 10000))
            cerebro.broker.setcommission(commission or cfg.get("commission", 0.0))

            # ----- locate CSV file ------------------------------------------------
            data_path = None
            for root in search_dirs:
                cand = next(root.glob(f"{symbol}*.csv"), None)
                if cand:
                    data_path = cand
                    break
            if data_path is None:
                raise FileNotFoundError(
                    f"Could not find CSV for {symbol} in: {', '.join(map(str, search_dirs))}"
                )

            tf = bt.TimeFrame.Minutes if "_m" in data_path.stem else bt.TimeFrame.Days
            comp = int(data_path.stem.split("_")[-1][:-1]) if "_m" in data_path.stem else 1
            fmt = "%Y-%m-%d %H:%M:%S" if tf is bt.TimeFrame.Minutes else "%Y-%m-%d"

            data = bt.feeds.GenericCSVData(
                dataname=str(data_path),
                dtformat=fmt,
                timeframe=tf,
                compression=comp,
                datetime=0,
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
                openinterest=-1,
                fromdate=start,
                todate=end,
            )
            cerebro.adddata(data)

            cerebro.addstrategy(StratCls)
            cerebro.addanalyzer(PositionSignal, _name="possig")

            res = cerebro.run()
            strat = res[0]
            ser: pd.Series = strat.analyzers.possig.get_analysis()
            ser.name = f"{StratCls.__name__}_signal"
            sig_frames.append(ser)

        # Merge on date index (outer join)
        df = pd.concat(sig_frames, axis=1).sort_index()
        df.index.name = "Date"

        outfile = outdir / f"{symbol}.csv"
        df.to_csv(outfile, float_format="%.0f")
        print(f"   saved → {outfile.relative_to(Path.cwd().resolve())}")


# --------------------------------------------------------------------------- 
# CLI
# --------------------------------------------------------------------------- 

def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Export per‑ticker signal CSVs from Backtrader strategies")
    parser.add_argument("--config", default="config.yaml", help="YAML config file (e.g. config.yaml)")
    parser.add_argument("--outdir", default="/signals", help="Destination folder for CSVs")
    parser.add_argument("--strats", nargs="*", help="Names of strategy classes to include (default: all)")
    parser.add_argument("--commission", type=float, help="Override commission (fraction)")
    parser.add_argument("--data-dirs", nargs="*", help="Extra folders to search for price CSVs (overrides defaults)")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    export_signals(
        cfg,
        outdir=args.outdir,
        strats_to_use=args.strats,
        commission=args.commission,
        data_dirs=args.data_dirs,
    )


if __name__ == "__main__":
    main()
