"""Monte Carlo out-of-sample evaluation for the Nova Quant Club leaderboard.

Instead of scoring one chained path through a few hand-picked windows, this
evaluates every strategy on many random *real* sub-windows drawn from a long
hidden history and ranks by the median (or another quantile) of the
per-window Sharpe ratios.

Fairness properties:
  * Same windows for every strategy (they depend only on the seed and the SPY
    calendar), so strategies are compared on identical market paths.
  * Windows are stratified across the whole pool (one random start per
    stratum), so coverage of regimes doesn't depend on luck.
  * Real market dynamics (trends, vol regimes) are kept; nothing is simulated.

Keep MC_SEED secret in any shared deployment: the windows are a pure function
of the seed. Rotate it each season.

The evaluator owns window selection, so callers only provide a strategy file;
the hidden windows cannot be overridden through the app.
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from backtester import (
    WARMUP_BARS,
    Backtester,
    load_trader_from_file,
    print_backtest_results,
)

# Full-history downloads shared across instances: ticker data is fetched once.
_CACHE: Dict[tuple, pd.DataFrame] = {}


class MonteCarloBacktester(Backtester):
    def __init__(
        self,
        *args,
        n_windows: int = 40,
        window_days: int = 252,
        pool_start: str = "2006-01-01",
        pool_end: str = "2026-01-01",  # exclusive; fixed so windows stay stable
        seed: int = 0,
        score_quantile: float = 0.5,  # 0.5 = median; 0.25 rewards robustness
        min_windows: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_windows = n_windows
        self.window_days = window_days
        self.pool_start = pool_start
        self.pool_end = pool_end
        self.seed = seed
        self.score_quantile = score_quantile
        self.min_windows = min_windows

    # ---- data: download each ticker's whole pool once, slice per window ----
    def _full(self, ticker: str) -> pd.DataFrame:
        t = ticker.strip().upper()
        # Extra year before the pool start so the first window has warm-up bars.
        start = (pd.Timestamp(self.pool_start) - pd.Timedelta(days=365)).strftime(
            "%Y-%m-%d"
        )
        key = (t, start, self.pool_end)
        if key not in _CACHE:
            _CACHE[key] = Backtester._download_single(self, t, start, self.pool_end)
        return _CACHE[key]

    def _download_single(self, ticker, start_date, end_date):
        df = self._full(ticker)
        mask = (df.index >= pd.Timestamp(start_date)) & (
            df.index < pd.Timestamp(end_date)
        )
        return df.loc[mask].copy()

    # ---- window sampling ----
    def _windows(self, tickers: List[str]) -> List[tuple]:
        """Seeded, stratified (start, end) dates, identical for every strategy."""
        cal = self._full(self.benchmark_ticker).index
        lo = max(int(cal.searchsorted(pd.Timestamp(self.pool_start))), WARMUP_BARS)
        hi = len(cal) - 1 - self.window_days  # last valid start position
        if hi <= lo:
            raise ValueError("Pool is too short for the requested window length.")

        rng = np.random.default_rng(self.seed)
        edges = np.linspace(lo, hi + 1, self.n_windows + 1).astype(int)
        starts = [
            int(rng.integers(edges[k], max(edges[k + 1], edges[k] + 1)))
            for k in range(self.n_windows)
        ]

        # Drop windows where a declared ticker has no data for the warm-up.
        first_date = max(self._full(t).index[0] for t in tickers)
        windows = [
            (cal[p], cal[p + self.window_days])
            for p in starts
            if cal[p - WARMUP_BARS] >= first_date
        ]
        if len(windows) < self.min_windows:
            raise ValueError(
                f"Only {len(windows)} usable windows: one of the tickers lacks "
                f"history before {first_date.date()}. Use tickers with longer history."
            )
        return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in windows]

    def _q(self, values) -> float:
        return float(np.quantile(values, self.score_quantile))

    def _fan(self, strat_paths, spy_paths) -> Dict:
        """Percentiles per trading day (rows = windows, cols = days), in dollars."""

        def band(paths, q):
            vals = np.percentile(paths, q, axis=0) * self.initial_cash
            # Day 0 is the starting capital, so every band fans out from one point.
            return [round(float(self.initial_cash), 2)] + [round(float(v), 2) for v in vals]

        return {
            "p5": band(strat_paths, 5),
            "p25": band(strat_paths, 25),
            "p50": band(strat_paths, 50),
            "p75": band(strat_paths, 75),
            "p95": band(strat_paths, 95),
            "spy_p50": band(spy_paths, 50),
            "windows": int(len(strat_paths)),
        }

    # ---- scoring ----
    def run_multi_window(self, strategy_filepath: str) -> Dict:
        tickers = [
            t.strip().upper() for t in getattr(load_trader_from_file(strategy_filepath), "tickers", [])
        ]
        if not tickers or len(tickers) > 20:
            raise ValueError("Strategy must declare between 1 and 20 valid yfinance tickers.")

        results = []
        for start, end in self._windows(tickers):
            trader = load_trader_from_file(strategy_filepath)  # fresh state per window
            results.append(self.run(trader, start, end))

        sharpes = np.array([r["sharpe"] for r in results])
        spy_sharpes = np.array([r["spy_sharpe"] for r in results])

        # Equity curve shown on the board: the window closest to the score.
        rep = results[int(np.argmin(np.abs(sharpes - self._q(sharpes))))]
        curve = rep["equity_df"][["Strategy", "SPY_Benchmark"]].reset_index(drop=True)
        curve.index.name = "Trading Day"

        # Day-by-day percentile bands of portfolio value across all windows.
        n = min(len(r["equity_df"]) for r in results)
        strat_paths = np.array(
            [r["equity_df"]["Strategy"].to_numpy()[:n] for r in results]
        ) / self.initial_cash
        spy_paths = np.array(
            [r["equity_df"]["SPY_Benchmark"].to_numpy()[:n] for r in results]
        ) / self.initial_cash

        return {
            "sharpe": self._q(sharpes),
            "spy_sharpe": self._q(spy_sharpes),
            "total_return_pct": self._q([r["total_return_pct"] for r in results]),
            "max_drawdown_pct": self._q([r["max_drawdown_pct"] for r in results]),
            "alpha_pct": self._q([r["alpha_pct"] for r in results]),
            "beta": self._q([r["beta"] for r in results]),
            "tickers": results[0]["tickers"],
            "windows_tested": len(results),
            "equity_curve_df": curve,
            "window_sharpes": [round(float(s), 3) for s in sharpes],
            "mc_fan": self._fan(strat_paths, spy_paths),
            # extras (not stored by app.py today)
            "sharpe_p25": float(np.quantile(sharpes, 0.25)),
            "sharpe_p75": float(np.quantile(sharpes, 0.75)),
            "pct_windows_beat_spy": float(np.mean(sharpes > spy_sharpes) * 100.0),
        }

    def run_benchmark_multi_window(self) -> Dict:
        """SPY buy-and-hold on the same windows (quantile across windows)."""
        sharpes, rets, dds, finals = [], [], [], []
        for start, end in self._windows([self.benchmark_ticker]):
            df = self._download_single(self.benchmark_ticker, start, end)
            shares = self.initial_cash / (float(df["Open"].iloc[0]) * (1 + self.commission_rate))
            m = self._calc_metrics(shares * df["Close"])
            sharpes.append(m["sharpe"])
            rets.append(m["total_return_pct"])
            dds.append(m["max_drawdown_pct"])
            finals.append(m["final_equity"])
        return {
            "sharpe": self._q(sharpes),
            "total_return_pct": self._q(rets),
            "max_drawdown_pct": self._q(dds),
            "final_equity": self._q(finals),
            "windows": len(sharpes),
            "window_sharpes": [round(float(s), 3) for s in sharpes],
        }


def mc_fan_chart(fan: Dict, trading_days_per_year: int = 252):
    """Fan chart of portfolio value across the hidden windows (matplotlib Figure).

    Each band is computed day by day across windows: the shaded areas span the
    25th-75th and 5th-95th percentiles, the solid line is the median, and the
    dashed line is the SPY buy-and-hold median.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    blue, orange = "#3b82f6", "#f97316"
    x = np.arange(len(fan["p50"])) / trading_days_per_year

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.fill_between(x, fan["p5"], fan["p95"], color=blue, alpha=0.15, lw=0, label="5th-95th percentile")
    ax.fill_between(x, fan["p25"], fan["p75"], color=blue, alpha=0.30, lw=0, label="25th-75th percentile")
    ax.plot(x, fan["p50"], color=blue, lw=2, label="Median")
    if fan.get("spy_p50"):
        ax.plot(x, fan["spy_p50"], color=orange, lw=1.6, ls="--", label="SPY median")
    ax.axhline(fan["p50"][0], color="gray", lw=0.8, ls=":")

    # End-of-window value labels on the right, nudged apart so they don't overlap.
    labels = [(fan["p95"][-1], blue), (fan["p50"][-1], blue), (fan["p5"][-1], blue)]
    if fan.get("spy_p50"):
        labels.append((fan["spy_p50"][-1], orange))
    labels.sort(key=lambda t: t[0])
    lo = min(fan["p5"] + fan.get("spy_p50", []))
    hi = max(fan["p95"])
    gap = 0.055 * (hi - lo)
    ys = [v for v, _ in labels]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (v, color), y in zip(labels, ys):
        ax.text(x[-1] + 0.02 * x[-1], y, f"${v:,.0f}", va="center", fontsize=9, color=color)

    ax.set_xlim(0, x[-1] * 1.12)
    ax.margins(y=0.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("Years into window")
    ax.set_title(f"Portfolio value across {fan.get('windows', '?')} hidden windows")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def from_env(**overrides) -> MonteCarloBacktester:
    """Build the official evaluator from environment variables."""
    cfg = dict(
        n_windows=int(os.getenv("MC_WINDOWS", "40")),
        window_days=int(os.getenv("MC_WINDOW_DAYS", "252")),
        pool_start=os.getenv("MC_POOL_START", "2006-01-01"),
        pool_end=os.getenv("MC_POOL_END", "2026-01-01"),
        seed=int(os.getenv("MC_SEED", "0")),
        score_quantile=float(os.getenv("MC_SCORE_QUANTILE", "0.5")),
        anonymize=True,
    )
    cfg.update(overrides)
    return MonteCarloBacktester(**cfg)


if __name__ == "__main__":
    # Local check: score distribution on random windows (random seed by default,
    # so this does NOT reveal the official windows).
    p = argparse.ArgumentParser(description="Monte Carlo window evaluation")
    p.add_argument("strategy_file")
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    seed = a.seed if a.seed is not None else int(time.time())
    res = from_env(n_windows=a.windows, seed=seed).run_multi_window(a.strategy_file)
    print(f"\nWindows used: {res['windows_tested']}  (seed {seed})")
    print_backtest_results({**res, "final_equity": res["equity_curve_df"]["Strategy"].iloc[-1]})
    print(f"Sharpe IQR              : {res['sharpe_p25']:.3f} .. {res['sharpe_p75']:.3f}")
    print(f"Windows beating SPY     : {res['pct_windows_beat_spy']:.0f}%")