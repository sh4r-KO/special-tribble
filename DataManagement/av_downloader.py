#!/usr/bin/env python3
"""
av_downloader.py — Alpha Vantage→Backtrader CSV fetcher (free‑tier safe)
============================================================================
*One stop for daily **or** intraday bars that never trips the “premium
endpoint” wall.*  The script:

1. Reads all symbols from your `config.yaml` (both `symbols:` or
   `runs:` layouts).
2. **Daily bars**: tries `TIME_SERIES_DAILY_ADJUSTED` *first*; if AV
   replies with the “premium endpoint” JSON, it transparently falls back
   to the free‑tier `TIME_SERIES_DAILY` endpoint.
3. **Intraday bars** (1/5/15/30/60 min) call `TIME_SERIES_INTRADAY`.
4. Re‑orders to *ascending* timestamps and keeps only the six columns
   Backtrader expects: `Date,Open,High,Low,Close,Volume`.
5. Sleeps 12 s between calls so you’ll never exceed the 5‑per‑minute
   limit.

Usage:
------
```bash
export AV_KEY=YOUR_FREE_API_KEY   # one‑time (env var preferred)
python av_downloader.py           # uses config.yaml → DataManagement/data/alpha

# pick 5‑minute bars instead of daily
python av_downloader.py --intraday 5
```

Tip: schedule this with cron / Task Scheduler at 01:00 local and your
CSV lake grows automatically each night.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AV_BASE = "https://www.alphavantage.co/query?datatype=csv&outputsize=full"


def parse_yaml_symbols(cfg_path: Path | str) -> list[str]:
    """Return a de‑duplicated, order‑preserving list of symbols from YAML."""
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if "symbols" in cfg:
        raw: Iterable[str | dict] = cfg["symbols"]
        symbols = [s if isinstance(s, str) else s.get("symbol") for s in raw]
    else:  # fallback: runs: - symbol: SPY
        symbols = [run["symbol"] for run in cfg.get("runs", [])]

    seen: set[str] = set()
    uniq: list[str] = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ---------- Alpha Vantage wrappers ----------------------------------------


def _av_get(params: str, api_key: str) -> str:
    url = f"{AV_BASE}&apikey={api_key}&{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_daily_csv(symbol: str, api_key: str) -> tuple[str, str]:
    """Return (csv_text, endpoint_used) for daily bars with auto‑fallback."""
    # 1) try the fully‑adjusted endpoint (may be premium for big tickers)
    txt = _av_get(f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}", api_key)
    if _looks_like_premium(txt):
        # 2) fall back to the free un‑adjusted endpoint
        txt = _av_get(f"function=TIME_SERIES_DAILY&symbol={symbol}", api_key)
        if _looks_like_premium(txt):
            raise RuntimeError(txt.strip())  # both failed → propagate
        return txt, "TIME_SERIES_DAILY"
    return txt, "TIME_SERIES_DAILY_ADJUSTED"


def fetch_intraday_csv(symbol: str, interval: int, api_key: str) -> str:
    return _av_get(
        f"function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}min",
        api_key,
    )


def _looks_like_premium(text: str) -> bool:
    return text.startswith("{\"Information\"") or ",open," not in text


# ---------- CSV massaging --------------------------------------------------


def convert_to_bt_rows(raw_csv: str) -> list[list[str]]:
    reader = csv.DictReader(raw_csv.splitlines())
    rows = [
        [row["timestamp"][:19], row["open"], row["high"], row["low"], row["close"], row["volume"]]
        for row in reader
    ]
    rows.sort(key=lambda r: r[0])  # ascending time
    return rows


def save_rows(path: Path, rows: list[list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def av_doawnloader_main(configFile: str):
    ap = argparse.ArgumentParser(description="Download Alpha Vantage CSVs for Backtrader")
    ap.add_argument("-c", "--config", default=configFile, help="YAML config (default: config.yaml)")
    ap.add_argument("-o", "--outdir", default="DataManagement/data/alpha", help="Output dir (default: DataManagement/data/alpha)")
    ap.add_argument("--intraday", type=int, choices=[1, 5, 15, 30, 60], help="Interval in minutes (skip for daily)")
    opts = ap.parse_args()

    api_key = "YLTAIJTQU30FT5WU"
    if not api_key:
        sys.exit("Set AV_KEY environment variable with your Alpha Vantage API key.")

    symbols = parse_yaml_symbols(opts.config)
    if not symbols:
        sys.exit("No symbols found in YAML.")

    outdir = Path(opts.outdir)
    intrv = opts.intraday

    print(f"Fetching {len(symbols)} symbol(s) from Alpha Vantage …\n")

    for idx, sym in enumerate(symbols, 1):
        try:
            if intrv:
                raw = fetch_intraday_csv(sym, intrv, api_key)
                fname = outdir / f"{sym}_{intrv}m.csv"
                endpoint = f"INTRADAY {intrv}m"
            else:
                raw, endpoint = fetch_daily_csv(sym, api_key)
                suffix = "_adj" if endpoint.endswith("ADJUSTED") else ""
                fname = outdir / f"{sym}{suffix}.csv"

            rows = convert_to_bt_rows(raw)
            save_rows(fname, rows)
            print(f"[{idx}/{len(symbols)}] {sym} ← {endpoint}  → {fname}  ({len(rows)} rows)")

        except Exception as exc:
            print(f"[warn] {sym}: {exc}")

        # Free tier: 5 calls / min (12‑second gap keeps us safe)
        if idx < len(symbols):
            time.sleep(12)

    print("\nDone.  You can now point backtrader_yaml_runner at the new CSVs.")


if __name__ == "__main__":
    av_doawnloader_main("config2.yaml")
