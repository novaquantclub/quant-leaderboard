"""Generate and freeze synthetic OHLCV rounds from public calibration data.

No historical evaluation window is replayed or resampled. Training observations
are reduced to model parameters before any synthetic prices are generated.
This module never loads or executes participant strategies.
"""

import argparse
import hashlib
import json
import re
import secrets
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from interest_rate import (
    INTEREST_RATE,
    INTEREST_RATE_FILE,
    MarketData,
    validate_interest_rate,
)

GENERATOR_VERSION = "synthetic-market-v2"
COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MIN_TRAINING_BARS = 120
MAX_VALUES = 15_000_000


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")  # noqa: TRY004 - uniform public validation contract
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


@dataclass(frozen=True)
class SyntheticConfig:
    paths: int = 30
    evaluation_bars: int = 252
    warmup_bars: int = 100

    def __post_init__(self):
        _integer(self.paths, "paths", 1, 256)
        _integer(self.evaluation_bars, "evaluation_bars", 10, 2520)
        _integer(self.warmup_bars, "warmup_bars", 20, 1000)
        for name in ("paths", "evaluation_bars", "warmup_bars"):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass
class MarketModel:
    symbols: tuple[str, ...]
    parameters: dict
    training_sha256: str

    @property
    def model_sha256(self):
        return _digest({"symbols": self.symbols, "parameters": self.parameters})


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
    calibrate_market(market)
    return MarketData(
        {symbol: frame.loc[:, list(COLUMNS)] for symbol, frame in market.items()},
        interest_rate=rate,
    )


def calibrate_market(training: Mapping[str, pd.DataFrame]) -> MarketModel:
    """Fit aggregate statistics from the exact public dataset given to teams.

    Symbols must share a complete, ordered calendar: filling missing data here
    would silently change the training sample or introduce future observations.
    """
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
    source_hash = hashlib.sha256(
        _json_bytes(
            {
                "symbols": symbols,
                "columns": COLUMNS,
                "index": [str(value) for value in reference],
            }
        )
    )
    for symbol in symbols:
        source_hash.update(frames[symbol].to_numpy(dtype="<f8").tobytes(order="C"))
    interest_rate = getattr(training, "interest_rate", None)
    if interest_rate is not None:
        validate_interest_rate(interest_rate, reference)
        source_hash.update(
            _json_bytes({"observation": INTEREST_RATE, "unit": "annualized percent"})
        )
        source_hash.update(interest_rate.to_numpy(dtype="<f8").tobytes())

    gaps, sessions, autoregression, wicks, volume_sigma = [], [], [], [], []
    for symbol in symbols:
        frame = frames[symbol]
        opening, high, low, close, volume = frame.to_numpy().T
        gaps.append(np.log(opening[1:] / close[:-1]))
        sessions.append(np.log(close[1:] / opening[1:]))
        returns = np.diff(np.log(close))
        variance = np.var(returns[:-1], ddof=1)
        phi = (
            np.cov(returns[:-1], returns[1:])[0, 1] / variance
            if variance > 1e-12
            else 0.0
        )
        autoregression.append(float(np.clip(phi, -0.25, 0.25)))
        upper = np.maximum(np.log(high / np.maximum(opening, close)), 1e-6)
        lower = np.maximum(np.log(np.minimum(opening, close) / low), 1e-6)
        wicks.append(
            [
                [
                    float(np.log(side).mean()),
                    float(np.clip(np.log(side).std(), 0.05, 1.0)),
                ]
                for side in (upper, lower)
            ]
        )
        volume_sigma.append(float(np.clip(np.std(np.log1p(volume), ddof=1), 0.05, 1.0)))

    # Joint overnight/intraday covariance retains relationships across assets,
    # rather than generating independent ETFs or copying real candle ratios.
    observations = np.column_stack(gaps + sessions)
    covariance = np.atleast_2d(np.cov(observations, rowvar=False))
    covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
    covariance += np.eye(len(covariance)) * 1e-10
    means = observations.mean(axis=0)
    means = np.clip(
        means, -0.15 * np.sqrt(np.diag(covariance)), 0.15 * np.sqrt(np.diag(covariance))
    )
    parameters = {
        "means": means.tolist(),
        "covariance": covariance.tolist(),
        "autoregression": autoregression,
        "wicks": wicks,
        "volume_sigma": volume_sigma,
    }
    if interest_rate is not None:
        rate = interest_rate.to_numpy(dtype=float)
        level = float(rate.mean())
        variance = np.var(rate[:-1], ddof=1)
        phi = (
            float(np.clip(np.cov(rate[:-1], rate[1:])[0, 1] / variance, 0, 0.9995))
            if variance > 1e-12
            else 0.0
        )
        residual = rate[1:] - (level + phi * (rate[:-1] - level))
        joint = np.column_stack([observations, residual])
        threshold = float(np.median(rate))
        covariances = []
        for mask in (rate[:-1] <= threshold, rate[:-1] > threshold):
            sample = joint[mask] if mask.sum() >= max(40, 2 * joint.shape[1]) else joint
            covariance = np.atleast_2d(np.cov(sample, rowvar=False))
            covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
            covariance += np.eye(len(covariance)) * 1e-10
            covariances.append(covariance.tolist())
        parameters["interest_rate"] = {
            "mean": level,
            "autoregression": phi,
            "threshold": threshold,
            "initial_quantiles": np.quantile(
                rate, [0.1, 0.25, 0.5, 0.75, 0.9]
            ).tolist(),
            "bounds": [
                float(min(0, rate.min())),
                float(min(100, max(1, rate.max() + rate.std()))),
            ],
            "joint_covariances": covariances,
        }
    return MarketModel(symbols, parameters, source_hash.hexdigest())


def _prices_hash(prices):
    return hashlib.sha256(
        np.asarray(prices, dtype="<f8").tobytes(order="C")
    ).hexdigest()


@dataclass
class SyntheticRound:
    """A frozen dataset, shared by every team regardless of declared tickers."""

    prices: np.ndarray  # path, symbol, bar, OHLCV
    manifest: dict
    interest_rates: np.ndarray | None = None

    @property
    def dataset_id(self):
        return _digest(self.manifest)

    def path(self, path_index, symbols=None):
        _integer(path_index, "path_index", 0, self.prices.shape[0] - 1)
        universe = self.manifest["symbols"]
        selected = list(universe) if symbols is None else list(symbols)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - set(universe)
        ):
            raise ValueError("Select unique symbols from the round's fixed universe.")
        config = self.manifest["config"]
        index = pd.RangeIndex(
            -config["warmup_bars"], config["evaluation_bars"], name="Step"
        )
        market = {
            symbol: pd.DataFrame(
                self.prices[path_index, universe.index(symbol)].copy(),
                index=index,
                columns=COLUMNS,
            )
            for symbol in selected
        }
        rate = (
            None
            if self.interest_rates is None
            else pd.Series(
                self.interest_rates[path_index].copy(), index=index, name=INTEREST_RATE
            )
        )
        return MarketData(market, interest_rate=rate)

    def save(self, destination, *, private_seed=None):
        """Publish a complete private bundle atomically, refusing to overwrite a round."""
        _validate_round(self)
        if private_seed is not None:
            _integer(private_seed, "seed", 0, 2**128 - 1)
            if _digest({"seed": str(private_seed)}) != self.manifest["seed_commitment"]:
                raise ValueError("The supplied private seed does not match this round.")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"A round already exists at {destination}.")
        temporary = Path(tempfile.mkdtemp(prefix=".synthetic-", dir=destination.parent))
        try:
            arrays = {"prices": self.prices}
            if self.interest_rates is not None:
                arrays["interest_rates"] = self.interest_rates
            np.savez_compressed(temporary / "prices.npz", **arrays)
            (temporary / "manifest.json").write_bytes(_json_bytes(self.manifest))
            if private_seed is not None:
                (temporary / "private_seed.json").write_bytes(
                    _json_bytes({"seed": str(private_seed)})
                )
            temporary.rename(destination)
        finally:
            if temporary.exists():
                if temporary.resolve().parent != destination.parent.resolve():
                    raise RuntimeError(
                        "Refusing to clean up outside the round directory."
                    )
                shutil.rmtree(temporary)

    @classmethod
    def load(cls, directory, *, expected_dataset_id=None):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        with np.load(directory / "prices.npz", allow_pickle=False) as archive:
            prices = archive["prices"]
            rate = archive.get("interest_rates")
        result = cls(prices, manifest, rate)
        _validate_round(result)
        if expected_dataset_id is not None and result.dataset_id != expected_dataset_id:
            raise ValueError(
                "Synthetic dataset does not match the committed dataset ID."
            )
        result.prices.setflags(write=False)
        if rate is not None:
            rate.setflags(write=False)
        return result


def _validate_round(round_data):
    manifest = round_data.manifest
    if manifest.get("generator_version") not in (
        "synthetic-market-v1",
        GENERATOR_VERSION,
    ):
        raise ValueError("Unsupported synthetic generator version.")
    config = SyntheticConfig(**manifest["config"])
    symbols = manifest["symbols"]
    if not symbols or len(symbols) > 20 or symbols != sorted(set(symbols)):
        raise ValueError("Invalid synthetic universe.")
    expected = (
        config.paths,
        len(symbols),
        config.warmup_bars + config.evaluation_bars,
        len(COLUMNS),
    )
    prices = round_data.prices
    if prices.size > MAX_VALUES or prices.shape != expected or prices.dtype.kind != "f":
        raise ValueError("Synthetic snapshot has an invalid shape or data type.")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(
            "Synthetic snapshot must contain finite, positive OHLCV values."
        )
    opening, high, low, close = [prices[..., i] for i in range(4)]
    if (high < np.maximum(opening, close)).any() or (
        low > np.minimum(opening, close)
    ).any():
        raise ValueError("Synthetic snapshot has invalid OHLC ordering.")
    if manifest.get("prices_sha256") != _prices_hash(prices):
        raise ValueError("Synthetic snapshot hash does not match its manifest.")
    rate = round_data.interest_rates
    if rate is None:
        if "interest_rate_sha256" in manifest:
            raise ValueError("Synthetic interest-rate observations are missing.")
    else:
        if (
            rate.shape != (config.paths, config.warmup_bars + config.evaluation_bars)
            or rate.dtype.kind != "f"
            or not np.isfinite(rate).all()
            or (np.abs(rate) > 100).any()
            or prices.size + rate.size > MAX_VALUES
        ):
            raise ValueError("Synthetic interest-rate observations are invalid.")
        if manifest.get("interest_rate_sha256") != _prices_hash(rate):
            raise ValueError(
                "Synthetic interest-rate hash does not match its manifest."
            )


def generate_round(model: MarketModel, seed: int, config=None) -> SyntheticRound:
    """Draw fresh paths, with no source observations available to the simulator.

    Student-t innovations, persistent calm/stress regimes, and a shared variance
    process provide tail events and volatility clustering. Their constants are
    part of GENERATOR_VERSION, not undisclosed knobs tuned after submissions.
    """
    _integer(seed, "seed", 0, 2**128 - 1)
    seed = int(seed)
    config = config if config is not None else SyntheticConfig()
    assets = len(model.symbols)
    bars = config.warmup_bars + config.evaluation_bars
    rate_model = model.parameters.get("interest_rate")
    if (
        not 1 <= assets <= 20
        or config.paths * bars * (assets * len(COLUMNS) + bool(rate_model)) > MAX_VALUES
    ):
        raise ValueError("Synthetic round exceeds the supported dataset size.")
    parameters = model.parameters
    means = np.asarray(parameters["means"])
    covariance = np.asarray(parameters["covariance"])
    factor = np.linalg.cholesky(covariance)
    scales = np.sqrt(np.diag(covariance))
    phi = np.asarray(parameters["autoregression"])
    wicks = np.asarray(parameters["wicks"])
    volume_sigma = np.asarray(parameters["volume_sigma"])
    prices = np.empty((config.paths, assets, bars, len(COLUMNS)), dtype=float)
    interest_rates = None
    rate_factors = []
    if rate_model is not None:
        interest_rates = np.empty((config.paths, bars), dtype=float)
        rate_factors = [
            np.linalg.cholesky(matrix) for matrix in rate_model["joint_covariances"]
        ]
    for path_index in range(config.paths):
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(path_index, 0))
        )
        previous_close = np.full(assets, 100.0)
        previous_return = means[:assets] + means[assets:]
        if rate_model is not None:
            previous_rate = float(rng.choice(rate_model["initial_quantiles"]))
        variance = 1.0
        stress = bool(rng.integers(2))
        for bar in range(bars):
            if rng.random() > 0.97:
                stress = not stress
            regime_scale = (1.75 if stress else 0.75) / np.sqrt((1.75**2 + 0.75**2) / 2)
            # One radial draw gives multivariate t innovations with covariance
            # matching the fitted Gaussian covariance before volatility scaling.
            if rate_model is not None:
                factor = rate_factors[int(previous_rate > rate_model["threshold"])]
            joint_innovation = (factor @ rng.standard_normal(len(factor))) * np.sqrt(
                5 / rng.chisquare(7)
            )
            innovation = joint_innovation[: 2 * assets]
            returns = means + np.sqrt(variance) * regime_scale * innovation
            returns[assets:] += phi * (
                previous_return - means[:assets] - means[assets:]
            )
            returns = np.clip(returns, -0.25, 0.25)
            opening = previous_close * np.exp(returns[:assets])
            close = opening * np.exp(returns[assets:])
            prices[path_index, :, bar, 0] = opening
            prices[path_index, :, bar, 3] = close
            previous_close = close
            previous_return = returns[:assets] + returns[assets:]
            if rate_model is not None:
                previous_rate = float(
                    np.clip(
                        rate_model["mean"]
                        + rate_model["autoregression"]
                        * (previous_rate - rate_model["mean"])
                        + joint_innovation[-1],
                        *rate_model["bounds"],
                    )
                )
                interest_rates[path_index, bar] = previous_rate
            standardized = innovation / scales
            variance = float(
                np.clip(
                    0.02 + 0.92 * variance + 0.06 * variance * np.mean(standardized**2),
                    0.15,
                    6,
                )
            )

        for asset in range(assets):
            candle_rng = np.random.default_rng(
                np.random.SeedSequence(seed, spawn_key=(path_index, 1, asset))
            )
            candle = prices[path_index, asset]
            extensions = np.minimum(
                np.exp(
                    candle_rng.normal(
                        wicks[asset, :, 0], wicks[asset, :, 1], size=(bars, 2)
                    )
                ),
                0.25,
            )
            candle[:, 1] = np.maximum(candle[:, 0], candle[:, 3]) * np.exp(
                extensions[:, 0]
            )
            candle[:, 2] = np.minimum(candle[:, 0], candle[:, 3]) * np.exp(
                -extensions[:, 1]
            )
            volume_rng = np.random.default_rng(
                np.random.SeedSequence(seed, spawn_key=(path_index, 2, asset))
            )
            # More active candles tend to have more volume; every volume is new.
            activity = (
                np.abs(np.log(candle[:, 3] / candle[:, 0])) / scales[assets + asset]
            )
            candle[:, 4] = np.exp(
                np.clip(
                    volume_sigma[asset]
                    * (volume_rng.standard_normal(bars) + 0.25 * activity),
                    -10,
                    10,
                )
            )
            candle[:, :4] *= 100.0 / candle[config.warmup_bars, 0]
            candle[:, 4] /= candle[
                config.warmup_bars - 20 : config.warmup_bars, 4
            ].mean()

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "symbols": list(model.symbols),
        "config": asdict(config),
        "training_sha256": model.training_sha256,
        "model_sha256": model.model_sha256,
        "seed_commitment": _digest({"seed": str(seed)}),
        "prices_sha256": _prices_hash(prices),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    if interest_rates is not None:
        manifest["interest_rate_sha256"] = _prices_hash(interest_rates)
    result = SyntheticRound(prices, manifest, interest_rates)
    _validate_round(result)
    result.prices.setflags(write=False)
    if interest_rates is not None:
        interest_rates.setflags(write=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Freeze a private synthetic competition dataset from public training CSVs."
    )
    parser.add_argument(
        "training_directory",
        type=Path,
        help="One SYMBOL.csv per asset; Date or Step in the first column, followed by OHLCV.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="New private directory; existing rounds are never overwritten.",
    )
    parser.add_argument("--paths", type=int, default=30)
    parser.add_argument(
        "--bars", type=int, default=252, help="Scored bars per independent path."
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        help="Explicit seed for public practice data or controlled regeneration. Omit for a fresh 128-bit seed.",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(128)
    try:
        training = load_training(args.training_directory)
        model = calibrate_market(training)
        round_data = generate_round(
            model, seed, SyntheticConfig(args.paths, args.bars, args.warmup)
        )
        round_data.save(args.output_directory, private_seed=seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot create synthetic round: {exc}\n")
    print(
        f"Saved {args.paths} synthetic paths for {', '.join(model.symbols)} to {args.output_directory}."
    )
    print(f"Dataset commitment: {round_data.dataset_id}")
    print(
        "Keep this entire directory private. It contains future prices and the final-test seed."
    )


if __name__ == "__main__":
    main()
