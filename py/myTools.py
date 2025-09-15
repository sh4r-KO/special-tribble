# loader.py  – tiny YAML helpers
import yaml
from pathlib import Path
from collections import OrderedDict
from yahooquery import Screener

# ────────────────────────────────────────────────────────────────
# 1. Symbols (already finished in your file)
# ────────────────────────────────────────────────────────────────
def load_symbols(yaml_path: str | Path = "config.yaml") -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    raw = cfg.get("symbols", [])
    return list(OrderedDict.fromkeys(map(str, raw)))   # dedupe + preserve order


# ────────────────────────────────────────────────────────────────
# 2. Single-value helpers
# ────────────────────────────────────────────────────────────────
def _load_cfg(yaml_path: str | Path) -> dict:
    """Internal shortcut so we don’t reopen the file five times if you
    decide to call several helpers in a row."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_startDate(yaml_path: str | Path = "config.yaml") -> str:
    """Return the back-test start date as *string* (ISO-8601)."""
    cfg = _load_cfg(yaml_path)
    return str(cfg.get("start", "2005-01-01"))


def load_endDate(yaml_path: str | Path = "config.yaml") -> str:
    """Return the back-test end date as *string* (ISO-8601)."""
    cfg = _load_cfg(yaml_path)
    return str(cfg.get("end", "2020-01-01"))


def load_capital(yaml_path: str | Path = "config.yaml") -> float:
    """Return starting cash (float)."""
    cfg = _load_cfg(yaml_path)
    return float(cfg.get("capital", 10_000))


def load_comission(yaml_path: str | Path = "config.yaml") -> float:
    """Return commission (fractional, e.g. 0.001 → 0.1 %)."""
    cfg = _load_cfg(yaml_path)
    return float(cfg.get("commission", 0.001))

def load_slippage(yaml_path: str | Path = "config.yaml") -> float:
    """Return slippage (fractional, e.g. 0.0005 → 0.05 %)."""
    cfg = _load_cfg(yaml_path)
    return float(cfg.get("slippage", 0.0005))

def load_output_csv(yaml_path: str | Path = "config.yaml") -> str:
    """Return the CSV filename for results."""
    cfg = _load_cfg(yaml_path)
    return str(cfg.get("output_csv", "results.csv"))


def load_strats(yaml_path: str | Path = "config.yaml") -> list[type]:
    """
    Return a list of Backtrader Strategy *classes* whose names appear under
    `strats:` in the YAML file.

        strats:
          - SmaCross
          - Rsi2
          - Rsi2        # duplicate → kept only once

    If the key is missing, empty, "all", or "*", we simply return every
    class found by strats.retall() – identical to the old behaviour.
    """
    from py.strats import retall as _all_strats                      # classes
    cfg  = _load_cfg(yaml_path)                                   # helper ②
    raw  = cfg.get("strats", [])                                  # list|str|…
    if raw in ("all", "*") or not raw:                            # no filter
        return _all_strats()

    # normalise to list of str
    if isinstance(raw, str):
        raw = [raw]

    # look-up table: class-name → class
    lut = {cls.__name__: cls for cls in _all_strats()}

    selected, seen = [], set()
    for name in raw:
        if not isinstance(name, str):
            raise TypeError("Entries under 'strats' must be strings.")
        cls = lut.get(name)
        if cls is None:
            raise ValueError(f"Strategy '{name}' not found in strats.py")
        if cls not in seen:                                       # de-dupe
            selected.append(cls)
            seen.add(cls)

    return selected


import yfinance as yf
from datetime import date, timedelta

def filter_available(tickers: list[str]) -> list[str]:
    """Return only those tickers for which Yahoo has at least one daily bar."""
    ok = []
    for tkr in tickers:
        try:
            df = yf.download(tkr, period="5d", progress=False, threads=False)
            if not df.empty:          # non-empty ⇒ ticker exists (today)
                ok.append(tkr)
        except Exception:
            pass                      # network hiccup or 4xx error → skip
    return ok

