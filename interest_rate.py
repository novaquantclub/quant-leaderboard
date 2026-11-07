"""Public interest-rate observations and their conservative availability calendar."""

import argparse
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

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


def align_interest_rate(source, asset_index):
    """Expose each effective-date value only after two federal business days.

    FRED DFF is current-vintage history. This availability rule is conservative
    about the daily release time; it does not reconstruct historical revisions.
    """
    if list(source.columns) != ["DFF"] or source.empty:
        raise ValueError("Provide dated FRED DFF observations.")
    dates = pd.DatetimeIndex(pd.to_datetime(source.index))
    if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
        raise ValueError("Source interest-rate dates must be unique and ordered.")
    publication_lag = 2 * CustomBusinessDay(calendar=USFederalHolidayCalendar())
    available = pd.Series(
        pd.to_numeric(source.DFF, errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex([date + publication_lag for date in dates]),
    ).dropna()
    available = available.loc[~available.index.duplicated(keep="last")].sort_index()
    calendar = pd.DatetimeIndex(pd.to_datetime(asset_index))
    # Forward filling uses only releases already available at that day's close.
    result = available.reindex(calendar, method="ffill")
    result.index, result.name = asset_index.copy(), INTEREST_RATE
    validate_interest_rate(result, asset_index)
    return result


def download_interest_rate(asset_index):
    if pd.api.types.is_numeric_dtype(asset_index.dtype):
        raise ValueError(
            "Downloading interest rates requires actual asset observation dates."
        )
    calendar = pd.DatetimeIndex(pd.to_datetime(asset_index))
    query = urlencode(
        {
            "id": "DFF",
            "cosd": str((calendar.min() - pd.Timedelta(days=30)).date()),
            "coed": str(calendar.max().date()),
        }
    )
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    with urlopen(url, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Interest-rate download exceeds the size limit.")
    source = pd.read_csv(io.BytesIO(raw), index_col=0, na_values=["."])
    return align_interest_rate(source, asset_index), {
        "series": "DFF",
        "source_url": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "unit": "annualized percent",
        "availability": "Effective date plus two US federal business days, then forward fill",
        "vintage": "Fixed current-vintage history; historical revisions are not reconstructed",
    }


def main():
    from prepare_training import save_training
    from synthetic_market import calibrate_market, load_training

    parser = argparse.ArgumentParser(
        description="Add observable interest rates to a new training version."
    )
    parser.add_argument("training_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    market = load_training(args.training_directory)
    if market.interest_rate is not None:
        parser.error("The input training dataset already includes interest rate.")
    rate, provenance = download_interest_rate(next(iter(market.values())).index)
    named_market = MarketData(market, interest_rate=rate)
    save_training(
        named_market,
        args.output_directory,
        source_info={
            "parent_training_sha256": calibrate_market(market).training_sha256,
            "interest_rate": provenance,
        },
    )
    print(
        f"Saved {len(rate)} available interest-rate observations with the existing competition names."
    )


if __name__ == "__main__":
    main()
