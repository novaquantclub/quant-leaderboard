"""One-time download of public training data; never used for final scoring."""

import argparse
from pathlib import Path

import pandas as pd

from synthetic_market import COLUMNS, calibrate_market


def main():
    parser = argparse.ArgumentParser(
        description="Download and freeze public daily OHLCV data."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive end date.")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "GLD"])
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error(
            "Output already exists. Use a new directory to avoid revising a published dataset."
        )
    if not pd.Timestamp(args.start) < pd.Timestamp(args.end):
        parser.error("The start date must precede the end date.")
    import yfinance as yf

    frames = {}
    for symbol in args.tickers:
        symbol = symbol.strip().upper()
        if symbol in frames:
            parser.error("Ticker symbols must be unique.")
        frame = yf.download(
            symbol, start=args.start, end=args.end, auto_adjust=True, progress=False
        )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frames[symbol] = frame.loc[:, list(COLUMNS)].dropna()
    common = next(iter(frames.values())).index
    for frame in frames.values():
        common = common.intersection(frame.index)
    frames = {
        symbol: frame.loc[common].sort_index() for symbol, frame in frames.items()
    }
    model = calibrate_market(frames)
    args.output_directory.mkdir(parents=True)
    for symbol, frame in frames.items():
        frame.to_csv(args.output_directory / f"{symbol}.csv", index_label="Date")
    print(f"Saved {len(common)} common bars for {', '.join(model.symbols)}.")
    print("Give this frozen dataset to every team. Final scoring never downloads data.")


if __name__ == "__main__":
    main()
