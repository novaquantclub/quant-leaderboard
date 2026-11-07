"""Public market schemas, calendar validation and CSV loading for local practice."""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MIN_TRAINING_BARS = 120
INTEREST_RATE = "interest rate"
INTEREST_RATE_FILE = "interest rate.csv"


class MarketData(dict):
    """Tradable asset frames plus an optional observed interest-rate series."""

    def __init__(self, assets, *, interest_rate=None):
        super().__init__(assets)
        self.interest_rate = interest_rate


def validate_interest_rate(series, index):
    if (
        not isinstance(series, pd.Series)
        or not series.index.equals(index)
        or not series.index.is_unique
        or not series.index.is_monotonic_increasing
        or series.index.hasnans
    ):
        raise ValueError(
            "Interest rate must share the complete ordered asset calendar."
        )
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.abs(values) > 100).any():
        raise ValueError(
            "Interest rate must contain finite annualized percentages between -100 and 100."
        )


def validate_market(training: Mapping[str, pd.DataFrame]):
    """Validate the public asset universe and its available macro observations."""
    if not 1 <= len(training) <= 20:
        raise ValueError("Provide a fixed universe of 1 to 20 symbols.")
    frames = {}
    for symbol, frame in training.items():
        if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 32:
            raise ValueError(
                "Symbols must be non-empty strings of at most 32 characters."
            )
        clean = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=_-]{1,32}", clean):
            raise ValueError(
                "Symbols may contain uppercase letters, numbers, '.', '^', '=', '_', and '-'."
            )
        if clean in frames:
            raise ValueError(f"Duplicate symbol after normalization: {clean}.")
        if not isinstance(frame, pd.DataFrame) or not set(COLUMNS).issubset(
            frame.columns
        ):
            raise ValueError(
                f"{clean} needs Open, High, Low, Close, and Volume columns."
            )
        if not frame.columns.is_unique:
            raise ValueError(f"{clean} has duplicate column names.")
        if len(frame) < MIN_TRAINING_BARS:
            raise ValueError(
                f"{clean} needs at least {MIN_TRAINING_BARS} training bars."
            )
        if (
            not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
            or frame.index.hasnans
        ):
            raise ValueError(
                f"{clean} needs a unique, increasing index without missing values."
            )
        values = frame.loc[:, list(COLUMNS)].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
        ):
            raise ValueError(
                f"{clean} needs finite, positive prices and non-negative volume."
            )
        opening, high, low, close = values[:, :4].T
        if (
            (high < np.maximum(opening, close))
            | (low > np.minimum(opening, close))
            | (low > high)
        ).any():
            raise ValueError(f"{clean} has invalid OHLC ordering.")
        frames[clean] = frame.loc[:, list(COLUMNS)].astype(float)

    symbols = tuple(sorted(frames))
    reference = frames[symbols[0]].index
    if any(not frames[symbol].index.equals(reference) for symbol in symbols):
        raise ValueError("All symbols must have the same complete training calendar.")
    interest_rate = getattr(training, "interest_rate", None)
    if interest_rate is not None:
        validate_interest_rate(interest_rate, reference)
    return MarketData(frames, interest_rate=interest_rate)


def load_training(directory):
    paths = sorted(
        path
        for path in Path(directory).glob("*.csv")
        if path.name != INTEREST_RATE_FILE
    )
    frames = {
        path.stem.strip().upper(): pd.read_csv(
            path, index_col=0, float_precision="round_trip"
        )
        for path in paths
    }
    if len(frames) != len(paths):
        raise ValueError("Duplicate training symbols.")
    rate_path = Path(directory) / INTEREST_RATE_FILE
    rate = None
    if rate_path.exists():
        data = pd.read_csv(rate_path, index_col=0, float_precision="round_trip")
        if list(data.columns) != [INTEREST_RATE]:
            raise ValueError(
                "The interest rate CSV needs one annualized percent column named 'interest rate'."
            )
        rate = data[INTEREST_RATE]
    market = MarketData(frames, interest_rate=rate)
    validate_market(market)
    return MarketData(
        {symbol: frame.loc[:, list(COLUMNS)] for symbol, frame in market.items()},
        interest_rate=rate,
    )
