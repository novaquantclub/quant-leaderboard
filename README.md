# Nova Quant Club Leaderboard

Nova Quant Club is a strategy competition where members build algorithmic trading
strategies and compare them on a shared backtesting engine. The Streamlit app
provides a leaderboard, an interactive sandbox, official submissions, and an
organizer admin panel.

## What is evaluated?

Every strategy is evaluated against an SPY buy-and-hold benchmark using:

- **Annualized Sharpe ratio** (the official ranking metric)
- Total return
- Maximum drawdown
- Annualized CAPM alpha
- Beta to SPY
- Equity curve and final portfolio value

Official submissions are run across multiple historical, out-of-sample market
windows. The windows are chained into one continuous equity curve, carrying the
ending portfolio value from one regime into the next. Evaluation windows are
configured by the organizer and are not exposed to competitors.

The default simulation assumptions are:

- Starting capital: **$100,000**
- Commission/slippage: **10 basis points (0.10%)**
- Risk-free rate: **4%**
- 252 trading days per year
- SPY benchmark purchased at the first available open, after the same fee

## Run locally

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, usually
`http://localhost:8501`.

The app creates its SQLite database and submission directory under `data/`.
Market data is downloaded through `yfinance`, so an internet connection is
required when running backtests.

## Build a strategy

Start with [`sample_strategy.py`](sample_strategy.py) or download it from the
Guide tab. A strategy file must define a `Trader` class with:

```python
class Trader:
    tickers = ["SPY", "QQQ", "GLD"]

    def __init__(self):
        self.name = "My Strategy"
        self.author = "My Name"

    def run(self, state):
        return {"SPY": 10}
```

### Strategy requirements

- `tickers` must contain between 1 and 20 valid Yahoo Finance ticker symbols.
- `name` and `author` must be non-empty strings.
- `run(state)` is called once per trading day.
- Return a dictionary mapping ticker symbols to share quantities. Positive
  quantities buy; negative quantities sell.
- Symbols containing `^` or `=` are read-only macro signals. They are available
  in `state.history`, but all non-zero orders for them are ignored.
- At evaluation `step == 0`, `state.history` already contains the prior 100
  trading bars plus the current bar. These warm-up bars are provided for
  indicators but are not scored or traded. In anonymized mode, the history
  index runs from `-100` through the current `step`, and evaluation bar `0`
  has an open price of 100.
- Orders returned at the current bar are filled at the **next bar's open**.
- Orders are subject to available cash, current positions, and the 10 bps fee.
- Strategies may optionally return `(orders, trader_data)`. `trader_data` must
  be a string smaller than 100 KB and can preserve state between bars.

The `state` object includes:

| Attribute | Description |
| --- | --- |
| `step` | An anonymized integer bar index |
| `history` | OHLCV data through the current bar for each declared ticker |
| `positions` | Current share positions by ticker |
| `cash` | Available cash |
| `portfolio_value` | Cash plus marked-to-market positions |
| `trader_data` | Persistent string state from the previous bar |

Prices and timestamps are anonymized during official evaluation: prices are
rebased to 100 at the first bar and dates are replaced with integer bar indices.
This helps prevent strategies from identifying hidden evaluation regimes.

## Test locally

Use the same backtesting engine as the web app:

```powershell
python backtester.py sample_strategy.py --start 2024-01-01 --end 2025-01-01
```

The web app's **Backtester** tab also accepts a `.py` file and plots the strategy
equity curve against SPY on selected in-sample dates. Local results are useful
for development, but they do not predict performance on the hidden official
windows.

## Submit to the leaderboard

1. Implement your `Trader` class and test it locally.
2. Open **Submit to Leaderboard**.
3. Enter the club upload password.
4. Upload the strategy `.py` file.
5. Review the scored metrics on the **Leaderboard** tab.

The leaderboard uses `Trader.author` and `Trader.name` as the member and
strategy identifiers. A member can update an existing member/strategy entry,
subject to the configured submission cooldown.

## Competition rules

1. **No lookahead bias:** `state.history` contains only data available through
   the current bar.
2. **Warm-up history:** each evaluation window starts at `step == 0` with 100
   prior trading bars in `state.history`; warm-up bars do not affect scoring or
   portfolio execution.
3. **Next-open execution:** orders are filled at the following day's open.
4. **Transaction costs apply:** every trade includes 10 bps commission/slippage.
5. **Read-only macro symbols:** symbols containing `^` or `=` may be used for
   signals but cannot be traded.
6. **Long-only positions:** sell orders cannot exceed the current position.
7. **No date leakage:** official timestamps and prices are anonymized.
8. **Benchmark consistency:** every score is compared with SPY under the same
   starting capital and fee assumptions.

## Configuration

The app supports these environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLUB_PASSWORD` | Password required for member submissions | `password` |
| `ADMIN_PASSWORD` | Password for organizer controls | `password` |
| `SECRET_OOS_WINDOWS` | Initial comma-separated `start:end` evaluation windows | Two example 2018 windows |
| `COOLDOWN_HOURS` | Minimum hours between updates to the same member/strategy | `0` |
| `DB_PATH` | SQLite database location | `data/leaderboard.db` |

For any shared or deployed instance, set strong passwords and configure
evaluation windows through environment variables or the Admin tab. The default
passwords are intended only for local development.

## Project structure

| File | Purpose |
| --- | --- |
| [`app.py`](app.py) | Streamlit UI, submissions, leaderboard, and admin controls |
| [`backtester.py`](backtester.py) | Data loading, simulation, metrics, and CLI |
| [`sample_strategy.py`](sample_strategy.py) | Starter moving-average strategy |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`nixpacks.toml`](nixpacks.toml) | Deployment start command |
| `data/` | Runtime SQLite database and uploaded submissions |

## Deployment

The repository includes a Nixpacks configuration that starts the app with:

```text
streamlit run app.py
```

Provide the environment variables above in the deployment platform, persist the
`data/` directory if leaderboard history must survive redeployments, and ensure
the service can reach Yahoo Finance.
