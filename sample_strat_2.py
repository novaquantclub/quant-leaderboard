"""Long-only ETF portfolio with fixed weights and periodic rebalancing."""

from math import isfinite


class Trader:
    author = "Luís Moreira"
    name = "sample_strat_2"
    # Broad US equities, Nasdaq-100 equities, US Treasuries, and gold.
    allocation = {"SPY": 0.30, "QQQ": 0.30, "IEF": 0.15, "GLD": 0.25}
    tickers = tuple(allocation)

    rebalance_days = 10
    invested_fraction = 0.98
    min_trade_fraction = 0.005

    def run(self, state):
        value = state.portfolio_value
        if state.step % self.rebalance_days or not isfinite(value) or value <= 0:
            return {}

        # Leave a cash buffer for next-open price moves and trading fees.
        budget = value * self.invested_fraction
        min_trade = value * self.min_trade_fraction
        orders = {}
        for ticker, weight in self.allocation.items():
            close = float(state.history[ticker]["Close"].iloc[-1])
            if not isfinite(close) or close <= 0:
                return {}
            current_units = state.positions.get(ticker, 0.0)
            target_units = budget * weight / close
            change = target_units - current_units
            if abs(change * close) >= min_trade:
                orders[ticker] = change
        return orders
