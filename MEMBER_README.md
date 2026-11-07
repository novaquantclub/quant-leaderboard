# Nova Quant League member starter

Build a trading strategy using the supplied histories. Research the assets and
discover their relationships. This pilot package is used to evaluate strategy
development before the competition rules are finalized.

## Your task

Read the data and the starter example, then create your own `strategy.py`.
Backtest it and record the research, validation periods, results, and exact
reproduction commands in `RESULTS.md`. Treat the supplied datasets and backtester
as fixed competition inputs. Choose chronological research and validation periods
and record which observations informed each development decision.

## Package contents

- `training/` contains one daily OHLCV CSV per asset and the supplied macro data.
- `sample_strategy.py` is a runnable example of the strategy interface.
- `backtester.py` and its support modules provide local backtesting.
- `requirements.txt` lists the available numerical libraries.
- `engine_config.json` records the public pilot execution and scoring settings.
- `nova-quant-member-starter.code-workspace` opens this folder in VS Code.

Local support includes market schema validation and strategy execution. Private
round preparation runs in the organizer's application.

The competition assets are {{ASSETS}}. Each asset file contains the following
columns on a shared trading calendar.

| Column | Meaning |
| --- | --- |
| `Date` | Trading date |
| `Open` | Adjusted opening price |
| `High` | Adjusted session high |
| `Low` | Adjusted session low |
| `Close` | Adjusted closing price |
| `Volume` | Shares traded |

Prices use consistent corporate-action adjustments. The CSVs retain their
historical price and volume scales. Local and hosted backtests normalize each
asset's first scored open to 100 and divide its volume by the mean of the final
20 warm-up observations. Quantities in a strategy refer to simulation units.

### interest rate

When supplied, `interest rate.csv` has `Date` and `interest rate` columns. Its values
are annualized percentages. For example, `5.25` means 5.25% per year. The series
is the effective federal funds rate, aligned to the asset calendar using the
latest observation available after a conservative two-US-federal-business-day
publication lag. The frozen history includes revisions available at download
time. Source details are available from FRED.
[FRED DFF](https://fred.stlouisfed.org/series/DFF)

The rate is a macroeconomic observation. Its values retain their percentage
units in the strategy API.

## Setup

Use Python 3.11 or newer. Open a terminal in this folder.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe backtester.py sample_strategy.py --data training --config engine_config.json --benchmark {{BENCHMARK}} --trusted-local
```

### macOS or Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python backtester.py sample_strategy.py --data training --config engine_config.json --benchmark {{BENCHMARK}} --trusted-local
```

Use the same command with `strategy.py` when testing your own entry. The
`--trusted-local` option executes the named Python file on your computer.

## Strategy interface

Your file defines a `Trader` class with the following members.

| Member | Contract |
| --- | --- |
| `name` | Non-empty strategy name, up to 120 characters |
| `author` | Non-empty author or team label, up to 120 characters |
| `tickers` | Sequence of unique uppercase asset names from the supplied universe |
| `run(state)` | Called once at each daily close, beginning at step 0 |

The existing starter shows how to declare these members. A fresh `Trader` is
created for each evaluation path. Strategies should be deterministic. Seed all
random generators explicitly, including independent NumPy generators and model
`random_state` arguments.

Each state contains the following fields.

| Field | Contents |
| --- | --- |
| `step` | Current integer step, beginning at 0 |
| `history` | Dictionary of OHLCV DataFrames for the declared assets, through the current step |
| `observations` | Dictionary of macro observation histories |
| `positions` | Current units held in each declared asset |
| `cash` | Available cash |
| `portfolio_value` | Cash plus the current marked value of positions |
| `trader_data` | Persistent string returned by the previous call, initially empty |

History includes {{WARMUP_BARS}} warm-up observations at negative integer indices.
At step 0, history runs from `-{{WARMUP_BARS}}` through `0`. Each subsequent call
adds the next available observation. Date-based research uses the CSVs, while
strategy execution uses these integer indices. The first
{{WARMUP_BARS}} CSV rows form the warm-up for a local backtest.

`state.observations["interest rate"]` is a pandas Series using the same available
integer index as asset history. Its latest value can be read with `.iloc[-1]`.
The macro history is supplied alongside the declared assets.

Return a dictionary mapping declared asset names to order quantities, or a tuple
of `(orders, trader_data)`. Positive quantities buy and negative quantities sell.
Each quantity is a change in units. For example, requesting 10 units when
already holding 5 produces a position of 15 after an unconstrained fill.
Quantities can be fractional and must be finite. Keep `trader_data` within
100 KB of UTF-8 text.

Hosted strategies run with the current state and installed numerical libraries.
Use that state for observations and persistent strategy data. The hosted
environment blocks network access and access to local research files. The
submission consists of one self-contained Python file up to 128 KB. Supporting
parameters can be included in that file.

Online practice is available while submissions are open. Local practice remains
available using this package. The hosted worker permits 1 GiB of address space,
30 CPU seconds, a 45-second lifetime per path, and three seconds per strategy
response. Startup has a 15-second allowance within that lifetime.

## Pilot execution and scoring

- Initial cash is {{INITIAL_CASH}}.
- Positions are long-only, financed by available cash. The engine caps sells at
  units held and buys at affordable quantities. Sells execute before buys.
- Orders decided at a close fill at the next open. The last close has no further
  fill opportunity within that path.
- Each executed order pays {{COMMISSION_BPS}} basis points on its traded value.
- Price formation follows the supplied bars. Volume is an observed feature.
- Cash earns zero interest in this pilot. The observed rate is available for
  strategy research.
- Both portfolios begin in cash at the first scored close. The comparison
  benchmark buys and holds {{BENCHMARK}} starting at the following open.
- Local output includes annualized Sharpe, return, maximum drawdown, final equity,
  and benchmark comparisons. Sharpe uses 252 periods per year and a fixed annual
  risk-free reference of {{RISK_FREE_RATE_PCT}}%.
- Final evaluation uses independent synthetic paths fitted to the public data.
  The pilot ranking uses median Sharpe across paths. The available macro context
  is generated together with asset prices.

## Reporting your results

In `RESULTS.md`, describe the strategy you developed, its assumptions, any
estimated parameters, research and validation periods, commands used, and the
reported metrics. Identify which results use observations that also informed
the strategy's design. Include runtime errors or limitations that affected your
evaluation. Select one self-contained `strategy.py` as your proposed entry.
