import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf


WARMUP_BARS = 100
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
OOS_JITTER_COPIES = 30
OOS_JITTER_SCALE = 0.10
READ_ONLY_SYMBOL_MARKERS = ("^", "=")


def is_read_only_symbol(symbol: str) -> bool:
    """Return whether a symbol may provide signals but cannot be traded."""
    return any(marker in symbol for marker in READ_ONLY_SYMBOL_MARKERS)


@dataclass
class TradingState:
    step: int  # Anonymized integer bar index (0, 1, 2, ...)
    history: Dict[str, pd.DataFrame]  # Indexed -100..step, prices normalized to base 100.0
    positions: Dict[str, float]  # Current units held per ticker
    cash: float
    portfolio_value: float
    trader_data: str = ""  # Persistent string state from previous bar


class Backtester:
    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission_bps: float = 10.0,
        risk_free_rate: float = 0.04,
        benchmark_ticker: str = "SPY",
        anonymize: bool = True,  # Scrubs calendar dates & rebases prices to 100
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_bps / 10_000.0
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker
        self.anonymize = anonymize

    def _download_single(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        t_clean = ticker.strip().upper()
        df = yf.download(
            t_clean, start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        if df.empty:
            raise ValueError(f"No data returned from yfinance for '{t_clean}'.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    def fetch_data(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        if not tickers or len(tickers) > 20:
            raise ValueError(
                "Strategy must declare between 1 and 20 valid yfinance tickers."
            )
        return {
            t.strip().upper(): self._download_single(t, start_date, end_date)
            for t in tickers
        }

    def _calc_metrics(self, equity_series: pd.Series, *, has_baseline=False) -> Dict:
        returns = equity_series.pct_change()
        returns = returns.iloc[1:] if has_baseline else returns.fillna(0.0)
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess = returns - daily_rf
        std_ret = excess.std()
        sharpe = (excess.mean() / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0.0

        total_return = (equity_series.iloc[-1] / self.initial_cash) - 1.0
        rolling_max = equity_series.cummax()
        max_dd = ((equity_series - rolling_max) / rolling_max).min()

        return {
            "sharpe": float(sharpe),
            "total_return_pct": float(total_return * 100.0),
            "max_drawdown_pct": float(max_dd * 100.0),
            "final_equity": float(equity_series.iloc[-1]),
            "returns": returns,
            "excess_returns": excess,
        }

    def run(self, trader, start_date: str, end_date: str) -> Dict:
        tickers = [t.strip().upper() for t in getattr(trader, "tickers", [])]
        if not tickers or len(tickers) > 20:
            raise ValueError(
                "Strategy must declare between 1 and 20 valid yfinance tickers."
            )
        evaluation_start = pd.Timestamp(start_date)
        evaluation_end = pd.Timestamp(end_date)
        buffer_start = evaluation_start - pd.Timedelta(days=365)
        raw_data = {
            ticker: self._download_single(
                ticker, buffer_start.strftime("%Y-%m-%d"), end_date
            )
            for ticker in tickers
        }
        spy_df = self._download_single(
            self.benchmark_ticker,
            buffer_start.strftime("%Y-%m-%d"),
            end_date,
        )

        # Use the benchmark's trading calendar as the master clock so assets
        # with broader calendars cannot introduce non-equity evaluation bars.
        available_dates = sorted(spy_df.index)
        evaluation_dates = [
            date for date in available_dates if evaluation_start <= date < evaluation_end
        ]
        warmup_dates = [
            date for date in available_dates if date < evaluation_start
        ][-WARMUP_BARS:]
        if len(evaluation_dates) < 10:
            raise ValueError("Insufficient trading days in selected window.")
        if len(warmup_dates) < WARMUP_BARS:
            raise ValueError(
                f"Insufficient history for the {WARMUP_BARS}-bar warm-up buffer."
            )
        return self._run_path(trader, raw_data, spy_df, warmup_dates, evaluation_dates)

    def _run_path(
        self, trader, raw_data, spy_df, warmup_dates, evaluation_dates, *, official=False
    ):
        tickers = list(raw_data)
        all_dates = warmup_dates + evaluation_dates
        warmup_end_index = len(warmup_dates)

        # Normalize evaluation bar 0 to 100 while retaining prior bars for warm-up.
        data = {}
        for t, df in raw_data.items():
            aligned = df.reindex(all_dates).ffill().bfill()
            if self.anonymize or official:
                base_price = float(aligned["Open"].iloc[warmup_end_index])
                base_vol = (
                    float(
                        aligned["Volume"]
                        .iloc[warmup_end_index - 20 : warmup_end_index]
                        .mean()
                    )
                    or 1.0
                )
                norm_df = aligned.copy()
                for col in PRICE_COLUMNS:
                    norm_df[col] = (aligned[col] / base_price) * 100.0
                norm_df["Volume"] = aligned["Volume"] / base_vol
                norm_df.index = pd.RangeIndex(
                    -warmup_end_index, len(norm_df) - warmup_end_index
                )
                data[t] = norm_df
            else:
                data[t] = aligned

        cash = self.initial_cash
        positions = {t: 0.0 for t in tickers}
        pending_orders: Dict[str, float] = {}
        equity_curve = []
        trader_data: str = ""

        for step_idx, dt in enumerate(evaluation_dates):
            data_index = warmup_end_index + step_idx

            # 1. Execute orders from previous bar at today's OPEN price
            for t, qty in sorted(pending_orders.items(), key=lambda x: x[1]):
                open_price = float(data[t].iloc[data_index]["Open"])
                if open_price <= 0 or np.isnan(open_price) or qty == 0:
                    continue
                if qty < 0:
                    qty = max(-positions[t], qty)
                elif qty > 0:
                    max_affordable = cash / (open_price * (1.0 + self.commission_rate))
                    qty = min(qty, max_affordable)

                trade_val = qty * open_price
                fee = abs(trade_val) * self.commission_rate
                cash -= trade_val + fee
                positions[t] += qty

            pending_orders = {}

            # 2. Mark-to-market at CLOSE
            port_val = cash
            current_history = {}
            for t in tickers:
                hist_slice = data[t].iloc[: data_index + 1]
                current_history[t] = hist_slice
                close_price = float(hist_slice["Close"].iloc[-1])
                port_val += positions[t] * close_price

            equity_curve.append({"Date": dt, "Strategy": port_val, "Cash": cash})

            # 3. Query Trader with scrubbed step index & normalized history
            state = TradingState(
                step=step_idx,
                history=current_history,
                positions=positions.copy(),
                cash=cash,
                portfolio_value=port_val,
                trader_data=trader_data,
            )

            result = trader.run(state)
            if isinstance(result, tuple) and len(result) == 2:
                orders, new_td = result
                if not isinstance(new_td, str) or len(new_td) > 100_000:
                    raise ValueError("trader_data must be a string under 100 KB.")
                trader_data = new_td
            else:
                orders = result

            if isinstance(orders, dict):
                pending_orders = {}
                for key, value in orders.items():
                    symbol = str(key).upper()
                    quantity = float(value)
                    if not np.isfinite(quantity) or quantity == 0:
                        continue
                    if symbol not in tickers:
                        continue
                    if is_read_only_symbol(symbol):
                        continue
                    pending_orders[symbol] = quantity

        eq_df = pd.DataFrame(equity_curve).set_index("Date")

        eq_df["SPY_Benchmark"] = self._benchmark_curve(spy_df.loc[evaluation_dates])

        if official:
            return eq_df

        return {
            **self._score_equity(eq_df),
            "equity_df": eq_df,
            "tickers": ", ".join(tickers),
        }

    def _score_equity(self, curve, *, has_baseline=False):
        strat_m = self._calc_metrics(curve["Strategy"], has_baseline=has_baseline)
        spy_m = self._calc_metrics(curve["SPY_Benchmark"], has_baseline=has_baseline)

        cov_matrix = np.cov(strat_m["excess_returns"], spy_m["excess_returns"])
        spy_var = cov_matrix[1, 1]
        beta = float(cov_matrix[0, 1] / spy_var) if spy_var > 1e-12 else 0.0
        daily_alpha = (
            strat_m["excess_returns"].mean() - beta * spy_m["excess_returns"].mean()
        )
        alpha_annual_pct = float(daily_alpha * 252 * 100.0)

        return {
            "sharpe": strat_m["sharpe"],
            "total_return_pct": strat_m["total_return_pct"],
            "max_drawdown_pct": strat_m["max_drawdown_pct"],
            "final_equity": strat_m["final_equity"],
            "spy_sharpe": spy_m["sharpe"],
            "spy_return_pct": spy_m["total_return_pct"],
            "spy_max_dd_pct": spy_m["max_drawdown_pct"],
            "spy_final_equity": spy_m["final_equity"],
            "alpha_pct": alpha_annual_pct,
            "beta": beta,
        }

    def _benchmark_curve(self, prices):
        shares = self.initial_cash / (float(prices["Open"].iloc[0]) * (1 + self.commission_rate))
        return shares * prices["Close"]

    def _jitter_paths(self, tickers, windows_str, seed):
        """Yield the same jittered prices for a symbol regardless of the strategy."""
        windows = sorted(
            tuple(pd.Timestamp(part.strip()) for part in entry.split(":"))
            for entry in windows_str.split(",") if entry.strip()
        )
        if not windows:
            raise ValueError("At least one historical source window is required.")
        for index, (start, end) in enumerate(windows):
            if start >= end:
                raise ValueError("Source window end dates must follow their start dates.")
            if index and start < windows[index - 1][1]:
                raise ValueError("Historical source windows must not overlap.")

        symbols = list(dict.fromkeys([self.benchmark_ticker, *tickers]))
        buffer_start = windows[0][0] - pd.Timedelta(days=365)
        raw_data = {
            symbol: self._download_single(
                symbol, buffer_start.strftime("%Y-%m-%d"), windows[-1][1].strftime("%Y-%m-%d")
            )
            for symbol in symbols
        }
        calendar = raw_data[self.benchmark_ticker].index

        for index, (start, end) in enumerate(windows):
            warmup = list(calendar[calendar < start][-WARMUP_BARS:])
            evaluation = list(calendar[(calendar >= start) & (calendar < end)])
            if len(warmup) < WARMUP_BARS or len(evaluation) < 10:
                raise ValueError(
                    f"Each window needs {WARMUP_BARS} prior SPY trading days and at least "
                    "10 SPY trading days to score."
                )
            dates = warmup + evaluation
            aligned = {
                symbol: raw_data[symbol].reindex(dates, method="ffill")
                for symbol in symbols
            }
            warmup_returns = {}
            for symbol, prices in aligned.items():
                values = prices[PRICE_COLUMNS].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(
                        f"'{symbol}' needs finite, positive prices throughout each window and its warm-up."
                    )
                warmup_returns[symbol] = np.diff(np.log(prices["Close"].iloc[:WARMUP_BARS]))

            # Share a market shock across all assets. Estimate its influence on
            # each symbol from the warm-up, then add symbol-specific noise.
            # Derive random streams from the window and symbol, not ticker order.
            shape = (OOS_JITTER_COPIES, len(evaluation))
            market = warmup_returns[self.benchmark_ticker]
            market_std = float(np.std(market, ddof=1))
            market_shock = np.random.default_rng([seed, index]).standard_normal(shape)
            levels = {}
            for symbol, returns in warmup_returns.items():
                if symbol == self.benchmark_ticker:
                    shocks = market_std * market_shock
                else:
                    loading = (
                        float(np.cov(returns, market)[0, 1] / market_std)
                        if market_std > 0 else 0.0
                    )
                    residual_std = np.sqrt(max(0.0, np.var(returns, ddof=1) - loading ** 2))
                    symbol_seed = int.from_bytes(symbol.encode("utf-8"), "little")
                    residual_shock = np.random.default_rng([seed, index, symbol_seed]).standard_normal(shape)
                    shocks = loading * market_shock + residual_std * residual_shock
                levels[symbol] = np.exp(np.cumsum(OOS_JITTER_SCALE * shocks, axis=1))

            for copy_index in range(OOS_JITTER_COPIES):
                jittered = {}
                for symbol, prices in aligned.items():
                    adjusted = prices.copy()
                    factors = np.concatenate((np.ones(WARMUP_BARS), levels[symbol][copy_index]))
                    adjusted[PRICE_COLUMNS] = prices[PRICE_COLUMNS].mul(factors, axis=0)
                    jittered[symbol] = adjusted
                yield copy_index, jittered, warmup, evaluation

    def _jittered_equity(self, tickers, windows_str, seed, strategy_filepath=None):
        # A copy uses the real dates within each window. Reset positions at
        # window boundaries, then compound the equity without adding gap returns.
        segments = [[] for _ in range(OOS_JITTER_COPIES)]
        for copy_index, prices, warmup, evaluation in self._jitter_paths(tickers, windows_str, seed):
            benchmark = prices[self.benchmark_ticker]
            if strategy_filepath is None:
                curve = self._benchmark_curve(benchmark.loc[evaluation]).to_frame("SPY_Benchmark")
            else:
                curve = self._run_path(
                    load_trader_from_file(strategy_filepath),
                    {symbol: prices[symbol] for symbol in tickers},
                    benchmark, warmup, evaluation, official=True,
                )[["Strategy", "SPY_Benchmark"]]
            segments[copy_index].append(curve.reset_index(drop=True))

        for copies in segments:
            baseline = pd.DataFrame(
                [[self.initial_cash] * len(copies[0].columns)], columns=copies[0].columns
            )
            parts = [baseline]
            for curve in copies:
                parts.append(curve * (parts[-1].iloc[-1] / self.initial_cash))
            yield pd.concat(parts, ignore_index=True)

    def run_jittered(self, strategy_filepath: str, windows_str: str, seed: int) -> Dict:
        """Rank a strategy by its average score on jittered historical prices."""
        trader = load_trader_from_file(strategy_filepath)
        tickers = [t.strip().upper() for t in getattr(trader, "tickers", [])]
        if not tickers or len(tickers) > 20:
            raise ValueError("Strategy must declare between 1 and 20 valid yfinance tickers.")

        curves = list(self._jittered_equity(tickers, windows_str, seed, strategy_filepath))
        results = [self._score_equity(curve, has_baseline=True) for curve in curves]
        metric_names = (
            "sharpe", "spy_sharpe", "total_return_pct", "max_drawdown_pct", "alpha_pct", "beta",
        )
        median_values = np.median([curve.to_numpy() for curve in curves], axis=0)
        median_equity = pd.DataFrame(median_values, columns=curves[0].columns)
        median_equity.index.name = "Trading Day"
        return {
            **{name: float(np.mean([result[name] for result in results])) for name in metric_names},
            "tickers": ", ".join(tickers),
            "paths_tested": len(results),
            "equity_curve_df": median_equity,
        }

    def run_benchmark_jittered(self, windows_str: str, seed: int) -> Dict:
        """Score SPY on the same jittered prices used for submissions."""
        results = [
            self._calc_metrics(curve["SPY_Benchmark"], has_baseline=True)
            for curve in self._jittered_equity([], windows_str, seed)
        ]
        return {
            name: float(np.mean([result[name] for result in results]))
            for name in ("sharpe", "total_return_pct", "max_drawdown_pct", "final_equity")
        }


def load_trader_from_file(filepath: str):
    spec = importlib.util.spec_from_file_location("user_strategy", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Trader"):
        raise AttributeError("Uploaded script must define a `Trader` class.")
    return module.Trader()


def print_backtest_results(results):
    print("\n--- Backtest Results ---")
    print(f"Annualized Sharpe Ratio : {results['sharpe']:.3f}")
    print(f"Total Return            : {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown            : {results['max_drawdown_pct']:.2f}%")
    print(f"Final Portfolio Equity  : ${results['final_equity']:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Club Local Backtester CLI")
    parser.add_argument(
        "strategy_file",
        help="Path to your Python strategy file (e.g., sample_strategy.py)",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"Loading strategy from {args.strategy_file}...")
    trader_instance = load_trader_from_file(args.strategy_file)
    bt = Backtester()
    print(
        f"Running backtest from {args.start} to {args.end} on tickers: {trader_instance.tickers}..."
    )
    results = bt.run(trader_instance, args.start, args.end)

    print_backtest_results(results)
