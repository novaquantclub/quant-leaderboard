"""One-time download of public training data; never used for final scoring."""

import argparse
import hashlib
import json
import os
import re
import secrets
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from interest_rate import INTEREST_RATE, INTEREST_RATE_FILE, MarketData
from synthetic_market import COLUMNS, calibrate_market


def repair_ohlc_roundoff(frame):
    """Repair only float64 rounding violations introduced by price adjustment."""
    frame = frame.copy()
    body_high = frame[["Open", "Close"]].max(axis=1)
    body_low = frame[["Open", "Close"]].min(axis=1)
    for column, bound, outside in (
        ("High", body_high, frame.High < body_high),
        ("Low", body_low, frame.Low > body_low),
    ):
        tolerance = 8 * np.spacing(bound.to_numpy(dtype=float))
        repair = outside & ((frame[column] - bound).abs() <= tolerance)
        frame.loc[repair, column] = bound[repair]
    return frame


def save_training(frames, destination, *, asset_names=None, source_info=None):
    """Save competition CSVs and separate organizer-only source provenance."""
    calibrate_market(frames)
    frames = MarketData(
        {symbol.strip().upper(): frame for symbol, frame in frames.items()},
        interest_rate=getattr(frames, "interest_rate", None),
    )
    symbols = sorted(frames)
    names = symbols.copy() if asset_names is None else list(asset_names)
    if asset_names is not None and (
        len(names) != len(symbols)
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", name) for name in names)
    ):
        raise ValueError("Provide one unique uppercase competition name per asset.")
    if asset_names is not None:
        secrets.SystemRandom().shuffle(names)
    public = {name: frames[symbol] for name, symbol in zip(names, symbols, strict=True)}
    destination = Path(destination).resolve()
    provenance_path = destination.with_name(f"{destination.name}.private.json")
    if destination.exists() or provenance_path.exists():
        raise FileExistsError("Training data or its private provenance already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".training-", dir=destination.parent
    ) as stage:
        public_dir = Path(stage) / "public"
        public_dir.mkdir()
        for name, frame in public.items():
            frame.loc[:, list(COLUMNS)].to_csv(
                public_dir / f"{name}.csv", index_label="Date"
            )
        if frames.interest_rate is not None:
            frames.interest_rate.rename(INTEREST_RATE).to_csv(
                public_dir / INTEREST_RATE_FILE, index_label="Date"
            )
        provenance = {
            "source": source_info or {},
            "source_tickers": dict(zip(names, symbols, strict=True)),
            "files_sha256": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(public_dir.glob("*.csv"))
            },
        }
        provenance_json = json.dumps(provenance, indent=2, allow_nan=False)
        descriptor = os.open(
            provenance_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(provenance_json)
            public_dir.rename(destination)
        except OSError:
            provenance_path.unlink()
            raise
    return tuple(sorted(public))


def main():
    parser = argparse.ArgumentParser(
        description="Download and freeze public daily OHLCV data."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive end date.")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "GLD"])
    parser.add_argument(
        "--asset-names",
        nargs="+",
        help="Neutral competition names, randomly assigned with a private mapping.",
    )
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
        frames[symbol] = repair_ohlc_roundoff(frame.loc[:, list(COLUMNS)].dropna())
    common = next(iter(frames.values())).index
    for frame in frames.values():
        common = common.intersection(frame.index)
    frames = {
        symbol: frame.loc[common].sort_index() for symbol, frame in frames.items()
    }
    names = save_training(
        frames,
        args.output_directory,
        asset_names=args.asset_names,
        source_info={
            "provider": "Yahoo Finance via yfinance",
            "start": args.start,
            "end_exclusive": args.end,
            "interval": "1d",
            "auto_adjust": True,
            "yfinance_version": yf.__version__,
            "ohlc_roundoff_repair": "Ordering violations of at most 8 float64 ULPs",
        },
    )
    print(f"Saved {len(common)} common bars for {', '.join(names)}.")
    print("Keep the sibling .private.json provenance file with the organizers.")
    print("Give this frozen dataset to every team. Final scoring never downloads data.")


if __name__ == "__main__":
    main()
