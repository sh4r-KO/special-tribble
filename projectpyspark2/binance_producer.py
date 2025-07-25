# binance_hourly_producer.py
import os, time, json, requests, logging
from confluent_kafka import Producer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
SYMBOLS   = ["BTCUSDT", "ETHUSDT"]           # Binance spot pairs
INTERVAL  = "1h"                             # hourly klines

URL = "https://api.binance.com/api/v3/klines"

# -----------------------------------------------------------------
# OLD
# producer = Producer({"bootstrap.servers": BOOTSTRAP,
#                      "enable.idempotence": True})

# NEW – extra reliability knobs
producer = Producer({
    "bootstrap.servers": BOOTSTRAP,
    "enable.idempotence": True,         # exactly‑once semantics
    "acks": "all",                      # wait for ISR replicas
    "request.timeout.ms": 30000,        # 30 s handshake window
    "max.in.flight.requests.per.connection": 5,
    "retries": 3                        # a few quick retries
})
# -----------------------------------------------------------------


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

def fetch_hour(symbol: str):
    params = {"symbol": symbol, "interval": INTERVAL, "limit": 1}
    r = requests.get(URL, params=params, timeout=5)
    r.raise_for_status()
    return r.json()[0]          # single kline

def main():
    while True:
        for sym in SYMBOLS:
            k = fetch_hour(sym)
            bar = {
                "t": k[0],      # openTime ms
                "o": float(k[1]),
                "h": float(k[2]),
                "l": float(k[3]),
                "c": float(k[4]),
                "v": float(k[5])
            }
            key = f"{sym}-{bar['t']}".encode()
            producer.produce("binance.bars", key=key, value=json.dumps(bar).encode())
            logging.info("Sent %s %s", sym, bar["t"])
        producer.flush()

        # sleep until the next full hour + 2 s buffer
        sleep_sec = 3600 - (time.time() % 3600) + 2
        time.sleep(sleep_sec)

if __name__ == "__main__":
    main()
