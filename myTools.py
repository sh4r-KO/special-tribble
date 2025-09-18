# loader.py  – tiny YAML helpers
import yaml
from pathlib import Path
from collections import OrderedDict
#from yahooquery import Screener
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Wedge, Circle
import pandas as pd
import mplfinance as mpf
import yfinance as yf
from datetime import date, timedelta
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
    return str(cfg.get("start"))


def load_endDate(yaml_path: str | Path = "config.yaml") -> str:
    """Return the back-test end date as *string* (ISO-8601)."""
    cfg = _load_cfg(yaml_path)
    return str(cfg.get("end"))


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
    from strats import retall as _all_strats                      # classes
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

def _find_local_csv(symbol: str, DATA_DIRS) -> Path | None:
    for root in DATA_DIRS:
        cand = next(root.glob(f"{symbol}*.csv"), None)
        if cand:
            return cand
    return None


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

def load_thresholds(path="DataManagement\data\PowerBi\Indicator_Target_Thresholds.csv") -> dict:
    df = pd.read_csv(path)
    thresholds = {}
    for _, row in df.iterrows():
        name = row["indicator"].strip()
        thresholds[name] = {
            "min": float(row["minimal target"]),
            "good": float(row["good target"]),
            "best": float(row["best target"]),
        }
    return thresholds


def gauge_percent(value, 
                  vmin=-1.0, vmax=0.0,           # range as fractions (-1 = -100%, 0 = 0%)
                  marker=None,                   # optional slim marker line
                  title="MaxDrawdown_%", 
                  as_percent=True,
                  color=["#556B2F","#8FA31E","#C6D870","#EFF5D2"]):
    """
    Draw a semi-circular gauge for a percentage-like metric.

    value:      current value (fraction, e.g., -0.39 for -39%).
                If you pass -39, set as_percent=False (and keep vmin/vmax consistent).
    vmin/vmax:  min/max of the scale (fractions if as_percent=True).
    marker:     draw a slim radial marker at this value (clamped for drawing, label shows raw).
    title:      text above the gauge.
    as_percent: if True, display with % signs. If False, show raw numbers.
    color:      list of 4 colors: [primary, secondary, third, fourth].
    """
    Cprimary, Csecond, Cthird, Cfourth = color

    # Keep the raw values for labeling, but clamp for drawing
    raw_value = value
    raw_marker = marker

    # Clamp for drawing on the dial
    def _clamp(x): 
        return min(max(x, vmin), vmax) if x is not None else None

    draw_value  = _clamp(value)
    draw_marker = _clamp(marker)

    def val2ang(v):
        """map value in [vmin,vmax] → angle in degrees [180,0]."""
        t = (v - vmin) / (vmax - vmin)
        return 180 * (1 - t)

    fig, ax = plt.subplots(figsize=(3.8, 2.6), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.4)
    ax.axis("off")

    # background ring
    base = Wedge((0,0), r=1.0, theta1=0, theta2=180, width=0.25,
                 facecolor=Cfourth, edgecolor="none")
    ax.add_patch(base)

    # main arc (use clamped value to stay within bounds)
    theta2 = val2ang(draw_value)
    arc = Wedge((0,0), r=1.0, theta1=theta2, theta2=180, width=0.25,
                facecolor=Csecond, edgecolor="none")
    ax.add_patch(arc)

    # needle (use clamped value)
    ang = math.radians(val2ang(draw_value))
    ax.plot([0.15*math.cos(ang), 1.05*math.cos(ang)],
            [0.15*math.sin(ang), 1.05*math.sin(ang)],
            lw=3, color=Cthird)
    ax.add_patch(Circle((0,0), radius=0.04, color=Cthird))

    # optional marker: draw at clamped position, label shows raw marker
    if marker is not None:
        ma = math.radians(val2ang(draw_marker))
        ax.plot([0.755*math.cos(ma), 1.12*math.cos(ma)],
                [0.755*math.sin(ma), 1.12*math.sin(ma)],
                lw=2, color="red")
        marker_label = f"{raw_marker:.2%}" if as_percent else f"{raw_marker:.2f}"
        ax.text(1.25*math.cos(ma), 1.25*math.sin(ma),
                marker_label, ha="center", va="center",
                fontsize=9, color="red")

    # ticks (min and max)
    for tv in (vmin, vmax):
        tick_label = f"{tv:.2%}" if as_percent else f"{tv:.2f}"
        ta = math.radians(val2ang(tv))
        ax.text(1.3*math.cos(ta), 1.2*math.sin(ta), tick_label,
                ha="center", va="center", fontsize=9, color=Cprimary)

    # center label shows the RAW value (even if outside range)
    center_label = f"{raw_value:.2%}" if as_percent else f"{raw_value:.2f}"
    # Optional: warn color if outside bounds
    if raw_value < vmin:
        center_color = "red"
    elif raw_value > vmax:
        center_color = "green"
    else:
        center_color = Cprimary

    ax.text(0, -0.3, center_label, ha="center", va="center", fontsize=26, color=center_color)
    ax.text(0, 1.37, title, ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    return fig, ax



def main():

    symbol = "AAPL"
    strat = "SmaCross"
    max_dd = 0.46502059193027323 # Example max drawdown values
    outdir = "output/graphs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    color = ["#19183B","#708993","#A1C2BD","#E7F2EF"]
    plot = gauge_percent(-1*max_dd,title=f"{symbol}_{strat}_MaxDrawdown", marker=-0.25, color=color)


    plot[0].savefig(f"{outdir}/{symbol}_{strat}_MaxDrawdown.png", dpi=300, bbox_inches="tight")
    plt.close(plot[0])  # free memory
    print(f"Saved: {outdir}/{symbol}_{strat}_MaxDrawdown.png")



    
if __name__ == "__main__":
    main()
