import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf


WARMUP_BARS = 100
READ_ONLY_SYMBOL_MARKERS = ("^", "=")


def is_read_only_symbol(symbol: str) -> bool:
    """Return whether a symbol may provide signals but cannot be traded."""
    return any(marker in symbol for marker in READ_ONLY_SYMBOL_MARKERS)


def chain_equity_curves(curves: List[pd.Series], initial_value: float) -> pd.Series:
    """Concatenate per-regime equity paths, carrying the ending level forward."""
    parts = []
    level = float(initial_value)
    for curve in curves:
        part = curve.astype(float) / float(curve.iloc[0]) * level
        level = float(part.iloc[-1])
        parts.append(part.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


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

    def _calc_metrics(self, equity_series: pd.Series) -> Dict[str, float]:
        returns = equity_series.pct_change().fillna(0.0)
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
        spy_df = self._download_single(self.benchmark_ticker, start_date, end_date)

        available_dates = sorted(
            list(set().union(*(df.index for df in raw_data.values())))
        )
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
        all_dates = warmup_dates + evaluation_dates
        warmup_end_index = len(warmup_dates)

        # Normalize evaluation bar 0 to 100 while retaining prior bars for warm-up.
        data = {}
        for t, df in raw_data.items():
            aligned = df.reindex(all_dates).ffill().bfill()
            if self.anonymize:
                base_price = float(aligned["Open"].iloc[warmup_end_index])
                base_vol = (
                    float(
                        aligned["Volume"]
                        .iloc[warmup_end_index : warmup_end_index + 5]
                        .mean()
                    )
                    or 1.0
                )
                norm_df = aligned.copy()
                for col in ["Open", "High", "Low", "Close"]:
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
            for t, qty in pending_orders.items():
                open_price = float(data[t].iloc[data_index]["Open"])
                if open_price <= 0 or np.isnan(open_price) or qty == 0:
                    continue
                if qty > 0:
                    max_affordable = cash / (open_price * (1.0 + self.commission_rate))
                    qty = min(qty, max_affordable)
                elif qty < 0:
                    qty = max(-positions[t], qty)

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
                    if symbol not in tickers:
                        continue
                    if quantity != 0 and is_read_only_symbol(symbol):
                        continue
                    pending_orders[symbol] = quantity

        eq_df = pd.DataFrame(equity_curve).set_index("Date")

        spy_aligned = spy_df["Close"].reindex(eq_df.index).ffill().bfill()
        spy_first_open = float(
            spy_df["Open"].reindex(eq_df.index).ffill().bfill().iloc[0]
        )
        spy_shares = self.initial_cash / (spy_first_open * (1.0 + self.commission_rate))
        eq_df["SPY_Benchmark"] = spy_shares * spy_aligned

        strat_m = self._calc_metrics(eq_df["Strategy"])
        spy_m = self._calc_metrics(eq_df["SPY_Benchmark"])

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
            "equity_df": eq_df,
            "tickers": ", ".join(tickers),
        }

    def run_multi_window(self, strategy_filepath: str, windows_str: str) -> Dict:
        """
        Evaluates a fresh Trader instance across multiple secret (start, end) regimes.
        windows_str format: "2018-09-01:2019-04-01,2022-01-01:2022-10-01,2025-02-01:2026-10-01"

        Regimes are stitched together sequentially: the ending equity of one
        regime becomes the starting equity of the next, giving one continuous
        portfolio path. All reported metrics are calculated from that combined
        path rather than averaged per regime.
        """
        pairs = [w.strip().split(":") for w in windows_str.split(",") if ":" in w]
        results = []
        for start_d, end_d in pairs:
            fresh_trader = load_trader_from_file(strategy_filepath)
            res = self.run(fresh_trader, start_d.strip(), end_d.strip())
            results.append(res)

        # Stitch each regime's path onto the end of the previous one. Every run
        # starts from flat cash, so we chain the relative performance and carry
        # the ending level forward. Indexes stay anonymized (bar numbers only).
        strategy_curve = chain_equity_curves(
            [res["equity_df"]["Strategy"] for res in results], self.initial_cash
        )
        spy_curve = chain_equity_curves(
            [res["equity_df"]["SPY_Benchmark"] for res in results], self.initial_cash
        )
        absolute_equity = pd.DataFrame(
            {"Strategy": strategy_curve, "SPY_Benchmark": spy_curve}
        )
        absolute_equity.index.name = "Trading Day"

        # Score the concatenated curve exactly like a single backtest.
        strat_m = self._calc_metrics(absolute_equity["Strategy"])
        spy_m = self._calc_metrics(absolute_equity["SPY_Benchmark"])

        cov_matrix = np.cov(strat_m["excess_returns"], spy_m["excess_returns"])
        spy_var = cov_matrix[1, 1]
        beta = float(cov_matrix[0, 1] / spy_var) if spy_var > 1e-12 else 0.0
        daily_alpha = (
            strat_m["excess_returns"].mean() - beta * spy_m["excess_returns"].mean()
        )
        alpha_annual_pct = float(daily_alpha * 252 * 100.0)

        equity_curve_df = absolute_equity.copy()
        equity_curve_df.index.name = "Trading Day"

        return {
            "sharpe": strat_m["sharpe"],
            "spy_sharpe": spy_m["sharpe"],
            "total_return_pct": strat_m["total_return_pct"],
            "max_drawdown_pct": strat_m["max_drawdown_pct"],
            "alpha_pct": alpha_annual_pct,
            "beta": beta,
            "tickers": results[0]["tickers"],
            "regimes_tested": len(results),
            "equity_curve_df": equity_curve_df,
        }

    def run_benchmark_multi_window(self, windows_str: str) -> Dict:
        """Buy-and-hold benchmark metrics on the same concatenated regime path."""
        pairs = [w.strip().split(":") for w in windows_str.split(",") if ":" in w]
        curves = []
        for start_d, end_d in pairs:
            spy_df = self._download_single(
                self.benchmark_ticker, start_d.strip(), end_d.strip()
            )
            spy_open = float(spy_df["Open"].iloc[0])
            shares = self.initial_cash / (spy_open * (1.0 + self.commission_rate))
            curves.append(shares * spy_df["Close"])

        combined = chain_equity_curves(curves, self.initial_cash)
        metrics = self._calc_metrics(combined)
        return {
            "sharpe": metrics["sharpe"],
            "total_return_pct": metrics["total_return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "final_equity": metrics["final_equity"],
            "regimes": len(pairs),
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
