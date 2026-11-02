import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TradingState:
    step: int  # Anonymized integer bar index (0, 1, 2, ...)
    history: Dict[str, pd.DataFrame]  # Indexed 0..step, prices normalized to base 100.0
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
        raw_data = self.fetch_data(tickers, start_date, end_date)
        spy_df = self._download_single(self.benchmark_ticker, start_date, end_date)

        all_dates = sorted(list(set().union(*(df.index for df in raw_data.values()))))
        if len(all_dates) < 10:
            raise ValueError("Insufficient trading days in selected window.")

        # Reindex and normalize each ticker to base 100 at bar 0 so strategies cannot fingerprint price levels
        data = {}
        for t, df in raw_data.items():
            aligned = df.reindex(all_dates).ffill().bfill()
            if self.anonymize:
                base_price = float(aligned["Open"].iloc[0])
                base_vol = float(aligned["Volume"].iloc[:5].mean()) or 1.0
                norm_df = aligned.copy()
                for col in ["Open", "High", "Low", "Close"]:
                    norm_df[col] = (aligned[col] / base_price) * 100.0
                norm_df["Volume"] = aligned["Volume"] / base_vol
                norm_df.index = pd.RangeIndex(len(norm_df))
                data[t] = norm_df
            else:
                data[t] = aligned

        cash = self.initial_cash
        positions = {t: 0.0 for t in tickers}
        pending_orders: Dict[str, float] = {}
        equity_curve = []
        trader_data: str = ""

        for step_idx, dt in enumerate(all_dates):
            idx_key = step_idx if self.anonymize else dt

            # 1. Execute orders from previous bar at today's OPEN price
            for t, qty in pending_orders.items():
                open_price = float(data[t].loc[idx_key, "Open"])
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
                hist_slice = data[t].iloc[: step_idx + 1]
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
                pending_orders = {
                    str(k).upper(): float(v)
                    for k, v in orders.items()
                    if str(k).upper() in tickers
                }

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
        """
        pairs = [w.strip().split(":") for w in windows_str.split(",") if ":" in w]
        results = []
        for start_d, end_d in pairs:
            fresh_trader = load_trader_from_file(strategy_filepath)
            res = self.run(fresh_trader, start_d.strip(), end_d.strip())
            results.append(res)

        # Base-100 equity curves per regime, indexed by anonymized bar number so
        # no calendar dates ever leave the backtester.
        regime_equity_curves = {}
        normalized_curves = []
        for regime_idx, res in enumerate(results, start=1):
            curve = res["equity_df"][["Strategy", "SPY_Benchmark"]].copy()
            curve = curve / self.initial_cash * 100.0
            curve.index = pd.RangeIndex(len(curve))
            regime_equity_curves[f"Regime {regime_idx}"] = curve
            normalized_curves.append(curve)

        # Average each bar across every regime. Only the bars covered by all
        # regimes are kept: otherwise the tail would silently drop the shorter
        # regimes and the "average" would jump toward the longest one's path.
        common_len = min(len(curve) for curve in normalized_curves)
        mean_equity_df = pd.DataFrame(
            {
                col: pd.concat(
                    [curve[col].iloc[:common_len] for curve in normalized_curves],
                    axis=1,
                ).mean(axis=1)
                for col in ("Strategy", "SPY_Benchmark")
            }
        )
        mean_equity_df.index.name = "Trading Day"

        # Robust aggregation across regimes: average metrics, penalize high variance across regimes
        sharpes = [r["sharpe"] for r in results]
        spy_sharpes = [r["spy_sharpe"] for r in results]
        return {
            "sharpe": float(np.mean(sharpes)),
            "min_regime_sharpe": float(np.min(sharpes)),
            "spy_sharpe": float(np.mean(spy_sharpes)),
            "total_return_pct": float(
                np.mean([r["total_return_pct"] for r in results])
            ),
            "max_drawdown_pct": float(np.min([r["max_drawdown_pct"] for r in results])),
            "alpha_pct": float(np.mean([r["alpha_pct"] for r in results])),
            "beta": float(np.mean([r["beta"] for r in results])),
            "tickers": results[0]["tickers"],
            "regimes_tested": len(results),
            "mean_equity_df": mean_equity_df,
            "regime_equity_curves": regime_equity_curves,
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
