import numpy as np


class Trader:
    # Use only symbols present in the round's published training CSVs.
    tickers = ("ATLAS", "BOREAL", "CIRRUS", "DUNE", "EMBER")

    def __init__(self):
        self.name = (
            "Simple Moving Average Crossover"  # Mandatory: give your strategy a name
        )
        self.author = "Your Name"  # Mandatory: give your name or alias

        self.short_window = 10
        self.long_window = 40

    def run(self, state) -> dict:
        """
        Called at the close of every trading day.
        Returns a dictionary of {ticker:shares_to_buy_or_sell} executed at next day's Open.

        Available state attributes:
        state.step              # Anonymized integer bar index (0, 1, 2, ...)
        state.history           # Indexed -100..step, prices normalized to base 100.0
        state.positions         # Current units held per ticker
        state.cash              # Current cash available for trading
        state.portfolio_value   # Current total portfolio value (cash + positions)
        state.trader_data       # Persistent string state from previous bar (max 100 KB)
        """
        orders = {}

        # Divide target capital equally across declared tickers
        target_allocation_per_ticker = state.portfolio_value / len(self.tickers)

        for ticker, df in state.history.items():
            if len(df) < self.long_window:
                continue

            closes = df["Close"]
            sma_short = closes.iloc[-self.short_window :].mean()
            sma_long = closes.iloc[-self.long_window :].mean()
            current_price = float(closes.iloc[-1])
            current_pos = state.positions.get(ticker, 0.0)

            # Golden cross: allocate target share weight
            if sma_short > sma_long:
                target_shares = np.floor(target_allocation_per_ticker / current_price)
                orders[ticker] = target_shares - current_pos
            # Death cross: exit position
            else:
                orders[ticker] = -current_pos

        return orders


if __name__ == "__main__":
    print(
        "Run: python backtester.py sample_strategy.py --data training --benchmark ATLAS --trusted-local"
    )
