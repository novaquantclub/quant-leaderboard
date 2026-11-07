"""Shared next-open engine. Web uploads use RemoteTrader, never Python imports."""

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_runner import RemoteTrader, validate_orders
from synthetic_market import COLUMNS, SyntheticRound, calibrate_market

ENGINE_VERSION = "streamed-next-open-v3"
WARMUP_BARS = 100


@dataclass(frozen=True)
class EngineConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 10.0
    risk_free_rate: float = 0.04
    benchmark: str = "SPY"
    warmup_bars: int = WARMUP_BARS
    runtime_id: str = "trusted-local"

    def __post_init__(self):
        if not np.isfinite(
            [self.initial_cash, self.commission_bps, self.risk_free_rate]
        ).all():
            raise ValueError("Engine settings must be finite.")
        if (
            self.initial_cash <= 0
            or not 0 <= self.commission_bps <= 1000
            or not -1 < self.risk_free_rate <= 1
        ):
            raise ValueError("Invalid cash, fee, or risk-free setting.")
        if not isinstance(self.warmup_bars, int) or not 20 <= self.warmup_bars <= 1000:
            raise ValueError("Warm-up must contain 20 to 1000 bars.")


@dataclass
class TradingState:
    step: int
    history: dict
    positions: dict
    cash: float
    portfolio_value: float
    trader_data: str = ""


def load_training(directory):
    paths = sorted(Path(directory).glob("*.csv"))
    frames = {
        path.stem.strip().upper(): pd.read_csv(
            path, index_col=0, float_precision="round_trip"
        )
        for path in paths
    }
    if len(frames) != len(paths):
        raise ValueError("Duplicate training symbols.")
    calibrate_market(frames)  # Same strict OHLCV/calendar validation as generation.
    return {symbol: frame.loc[:, list(COLUMNS)] for symbol, frame in frames.items()}


def evaluation_id(dataset_id, config):
    settings = {
        "engine": ENGINE_VERSION,
        "engine_code": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "dataset_id": dataset_id,
        "settings": asdict(config),
        "ranking": "median-sharpe",
        "interval": "conditional-bootstrap-median-2000-v1",
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class Backtester:
    def __init__(self, config=None):
        self.config = config or EngineConfig()

    def run_dataset(self, trader, market):
        config = self.config
        tickers = list(trader.tickers)
        if not tickers or len(tickers) > 20 or set(tickers) - set(market):
            raise ValueError(
                "Strategy tickers must belong to this round's fixed asset universe."
            )
        if config.benchmark not in market:
            raise ValueError("The benchmark must belong to the round's asset universe.")
        length = len(market[config.benchmark])
        if length < config.warmup_bars + 10:
            raise ValueError("Insufficient warm-up or evaluation data.")
        if any(
            len(frame) != length
            or not frame.index.equals(market[config.benchmark].index)
            for frame in market.values()
        ):
            raise ValueError("Every asset must share the same complete calendar.")
        data = {}
        for symbol in set(tickers) | {config.benchmark}:
            frame = market[symbol].loc[:, list(COLUMNS)].copy()
            if len(frame) != length or not frame.index.equals(
                market[config.benchmark].index
            ):
                raise ValueError(
                    "Every asset must share a complete calendar; data is never backfilled."
                )
            if (
                not np.isfinite(frame.to_numpy()).all()
                or (frame.iloc[:, :4].to_numpy() <= 0).any()
            ):
                raise ValueError("Invalid market prices.")
            frame.index = pd.RangeIndex(
                -config.warmup_bars, length - config.warmup_bars
            )
            frame.loc[:, list(COLUMNS[:4])] *= 100 / frame.loc[0, "Open"]
            base_volume = frame.loc[-20:-1, "Volume"].mean()
            frame["Volume"] /= base_volume if base_volume > 0 else 1
            data[symbol] = frame
        cash, positions, pending = config.initial_cash, dict.fromkeys(tickers, 0.0), {}
        trader_data, curve, benchmark_units = "", [], 0.0
        fee_rate = config.commission_bps / 10_000
        for step in range(length - config.warmup_bars):
            for symbol, quantity in sorted(pending.items(), key=lambda order: order[1]):
                if any(marker in symbol for marker in ("^", "=")):
                    continue
                opening = float(data[symbol].loc[step, "Open"])
                quantity = (
                    max(-positions[symbol], quantity)
                    if quantity < 0
                    else min(quantity, max(0.0, cash) / (opening * (1 + fee_rate)))
                )
                value = quantity * opening
                cash -= value + abs(value) * fee_rate
                positions[symbol] += quantity
            # Both portfolios start in cash at the first close. The benchmark's
            # first purchase executes at the same next open available to a team.
            if step == 1:
                benchmark_units = config.initial_cash / (
                    data[config.benchmark].loc[step, "Open"] * (1 + fee_rate)
                )
            benchmark = (
                float(benchmark_units * data[config.benchmark].loc[step, "Close"])
                if step
                else config.initial_cash
            )
            portfolio = cash + sum(
                positions[t] * float(data[t].loc[step, "Close"]) for t in tickers
            )
            curve.append((portfolio, benchmark))
            history = {
                symbol: data[symbol].loc[:step].copy(deep=True) for symbol in tickers
            }
            state = TradingState(
                step,
                history,
                positions.copy(),
                float(cash),
                float(portfolio),
                trader_data,
            )
            result = trader.run(state)
            orders, trader_data = (
                result
                if isinstance(result, tuple) and len(result) == 2
                else (result, trader_data)
            )
            pending, trader_data = validate_orders(
                {"orders": orders, "trader_data": trader_data}, tickers
            )
        benchmark_column = "SPY_Benchmark" if config.benchmark == "SPY" else "Benchmark"
        result = pd.DataFrame(curve, columns=["Strategy", benchmark_column])
        result.index.name = "Trading Day"
        return {
            **self._score_equity(result),
            "equity_df": result,
            "tickers": ", ".join(tickers),
        }

    def _calc_metrics(self, series):
        returns = series.pct_change().iloc[1:]
        daily_rf = (1 + self.config.risk_free_rate) ** (1 / 252) - 1
        excess = returns - daily_rf
        deviation = excess.std()
        return {
            "sharpe": float(excess.mean() / deviation * np.sqrt(252))
            if deviation > 1e-9
            else 0.0,
            "total_return_pct": float((series.iloc[-1] / series.iloc[0] - 1) * 100),
            "max_drawdown_pct": float((series / series.cummax() - 1).min() * 100),
            "final_equity": float(series.iloc[-1]),
            "excess_returns": excess,
        }

    def _score_equity(self, curve):
        strategy = self._calc_metrics(curve.Strategy)
        benchmark_column = (
            "SPY_Benchmark" if self.config.benchmark == "SPY" else "Benchmark"
        )
        benchmark = self._calc_metrics(curve[benchmark_column])
        covariance = np.cov(strategy["excess_returns"], benchmark["excess_returns"])
        beta = (
            float(covariance[0, 1] / covariance[1, 1])
            if covariance[1, 1] > 1e-12
            else 0.0
        )
        alpha = (
            (
                strategy["excess_returns"].mean()
                - beta * benchmark["excess_returns"].mean()
            )
            * 252
            * 100
        )
        benchmark_prefix = "spy" if self.config.benchmark == "SPY" else "benchmark"
        return {
            **{
                key: value for key, value in strategy.items() if key != "excess_returns"
            },
            f"{benchmark_prefix}_sharpe": benchmark["sharpe"],
            f"{benchmark_prefix}_return_pct": benchmark["total_return_pct"],
            f"{benchmark_prefix}_max_dd_pct": benchmark["max_drawdown_pct"],
            f"{benchmark_prefix}_final_equity": benchmark["final_equity"],
            "alpha_pct": float(alpha),
            "beta": beta,
        }

    def run_public(self, source, market, *, trader_factory=RemoteTrader):
        with trader_factory(source) as trader:
            return self.run_dataset(trader, market)

    def run_synthetic(
        self, source, round_data: SyntheticRound, *, trader_factory=RemoteTrader
    ):
        if round_data.manifest["config"]["warmup_bars"] != self.config.warmup_bars:
            raise ValueError("Engine and snapshot warm-up settings differ.")
        curves, metrics = [], []
        for path_index in range(round_data.manifest["config"]["paths"]):
            with trader_factory(source) as trader:
                result = self.run_dataset(trader, round_data.path(path_index))
            curves.append(result.pop("equity_df"))
            result.pop("tickers")
            metrics.append(result)
        values = np.array([result["sharpe"] for result in metrics])
        bootstrap = np.random.default_rng(0).choice(values, size=(2000, len(values)))
        confidence = np.quantile(np.median(bootstrap, axis=1), [0.025, 0.975])
        curve = pd.DataFrame(
            np.median([frame.to_numpy() for frame in curves], axis=0),
            columns=curves[0].columns,
        )
        return {
            **{
                name: float(np.median([row[name] for row in metrics]))
                for name in metrics[0]
            },
            "score": float(np.median(values)),
            "score_ci_low": float(confidence[0]),
            "score_ci_high": float(confidence[1]),
            "sharpe_p10": float(np.quantile(values, 0.1)),
            "sharpe_p90": float(np.quantile(values, 0.9)),
            "paths_tested": len(values),
            "path_sharpes": values.tolist(),
            "equity_curve": curve.to_dict(orient="list"),
            "evaluation_id": evaluation_id(round_data.dataset_id, self.config),
        }


def load_local_trader(filepath):
    """For the explicit trusted-local CLI only. Never use for web uploads."""
    spec = importlib.util.spec_from_file_location("local_strategy", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader()


def main():
    parser = argparse.ArgumentParser(
        description="Backtest your own strategy against the published training CSVs."
    )
    parser.add_argument("strategy_file")
    parser.add_argument("--data", default="data/training")
    parser.add_argument(
        "--benchmark", default="SPY", help="Benchmark asset in the dataset."
    )
    parser.add_argument(
        "--trusted-local",
        action="store_true",
        help="Execute your own code locally without the hosted worker.",
    )
    args = parser.parse_args()
    market = load_training(args.data)
    engine = Backtester(EngineConfig(benchmark=args.benchmark))
    if args.trusted_local:
        result = engine.run_dataset(load_local_trader(args.strategy_file), market)
    else:
        result = engine.run_public(
            Path(args.strategy_file).read_text(encoding="utf-8-sig"), market
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "equity_df"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
