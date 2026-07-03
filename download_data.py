"""
download_data.py

Helper script to reproduce the OHLCV dataset used in this study.
Downloads 15-minute, 4-hour, and daily klines for BNB/USDT and BTC/USDT
from the Binance public API and stores them as Parquet files.

Usage:
    python download_data.py --data-dir ~/crypto_data
"""
import os
import sys
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

API = "https://api.binance.com/api/v3/klines"

TICKERS = {
    "bnb": "BNBUSDT",
    "btc": "BTCUSDT",
}

TIMEFRAMES = {
    "15m": {"interval": "15m", "ms": 15 * 60 * 1000},
    "4h": {"interval": "4h", "ms": 4 * 60 * 60 * 1000},
    "1d": {"interval": "1d", "ms": 24 * 60 * 60 * 1000},
}

START_DATE = "2020-01-01"


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    """Fetch klines from Binance public API with pagination."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    rows = []
    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        start_ms = int(data[-1][0]) + 1
    df = pd.DataFrame(rows, columns=cols)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base", "taker_buy_quote"]
    df[numeric] = df[numeric].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["n_trades"] = df["n_trades"].astype(int)
    return df


def download_pair(ticker: str, symbol: str, tf_name: str, info: dict, data_dir: Path):
    out_path = data_dir / f"{ticker}_{tf_name}_usdt_binance.parquet"
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping")
        return

    start_ms = int(datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"Downloading {symbol} {info['interval']} ...")
    df = fetch_klines(symbol, info["interval"], start_ms, end_ms)
    df.to_parquet(out_path, index=False)
    print(f"  saved {out_path.name} ({len(df)} rows, {df['open_time'].min()} to {df['open_time'].max()})")


def main():
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data for reproduction")
    parser.add_argument("--data-dir", type=str, default="~/crypto_data",
                        help="Directory where Parquet files will be stored")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    for ticker, symbol in TICKERS.items():
        for tf_name, info in TIMEFRAMES.items():
            download_pair(ticker, symbol, tf_name, info, data_dir)

    print("\nDone. To verify, run:")
    print(f"  ls -lh {data_dir}")


if __name__ == "__main__":
    main()
