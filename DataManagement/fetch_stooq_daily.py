
#TODO change to get more symbols, this is pathetic
import pandas_datareader.data as web, pathlib, sys

def import_stooq():
    syms = sys.argv[1:] or ["SPY", "AAPL", "QQQ"]     # read from YAML later
    out = pathlib.Path(r"DataManagement/data/stooq"); out.mkdir(parents=True, exist_ok=True)

    for s in syms:
        df = web.DataReader("SPY", "stooq")              # full daily history
        df.to_csv(out / f"{s}_d.csv")
        print(f"Saved {s}  →  {out}/{s}_d.csv  ({len(df)} rows)")
