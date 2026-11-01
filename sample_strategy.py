import numpy as np


class Trader:
    # Any valid yfinance ticker(s)
    tickers = ["SPY", "QQQ", "GLD"]

    def __init__(self):
        self.name = "Simple Moving Average Crossover"
        self.author = "Your Name"
        
        self.short_window = 10
        self.long_window = 40

    def run(self, state) -> dict:
        """
        Called at the close of every trading day.
        Returns a dictionary of {ticker:shares_to_buy_or_sell} executed at next day's Open.
        """
        orders = {}
        # Divide target capital equally across declared tickers
        target_allocation_per_ticker = state.portfolio_value / len(self.tickers)

        for ticker, df in state.history.items():
            if len(df) < self.long_window:
                continue

            closes = df["Close"]
            sma_short = closes.iloc[-self.short_window:].mean()
            sma_long = closes.iloc[-self.long_window:].mean()
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