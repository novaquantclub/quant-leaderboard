# Synthetic competition datasets

`synthetic_market.py` prepares fixed synthetic datasets for the competition.
The Streamlit app still uses the historical jitter evaluator until the next
integration step. Preparing synthetic prices alone does not isolate uploaded
code or prevent lookahead inside the current evaluator.

## What the generator does

Give teams a public training dataset and use that same dataset to calibrate the
generator. All assets must have the same complete, ordered calendar and at least
120 bars. Calibration estimates joint overnight/intraday return covariance,
bounded return autocorrelation, candle extensions, and volume dispersion.

The simulator receives only these aggregate parameters. It draws new prices
using correlated Student-t innovations, persistent calm/stress regimes, and a
shared changing variance. Every warm-up bar, open, close, high, low, and volume
is synthetic. It does not replay a secret historical window, reuse historical
candles, or sample blocks of real returns. Prices have valid OHLC ordering and
the first scored open is rebased to 100; that is a single scale change across
the entire path, without a separate reset at the warm-up boundary. Volume is
normalized by the final 20 warm-up bars.

Every round uses one fixed asset universe. Generate the full universe before
reading any team's ticker declarations. Selecting fewer tickers later does not
change the paths for those tickers. Each path has an independent random stream;
paths are not noisy copies of a shared historical return sequence. Extending
the path count does not alter the existing paths.

This is a model-based competition. It tests performance under the published
generator assumptions, not a claim about actual future market returns. The
model's relationships, tails, regime persistence, and volatility clustering
should be checked on the club's actual training dataset before a round starts.
The generator constants are public and versioned; only final random draws and
the final seed are hidden. Independent synthetic paths do not remove model
misspecification or justify claims of real-world statistical significance.

## Prepare one round

Place the exact public training CSVs in `data/training/`, one file per asset,
such as `SPY.csv`, `QQQ.csv`, and `GLD.csv`. Each file needs an ordered `Date` or
`Step` column first, then `Open,High,Low,Close,Volume`.

```powershell
python synthetic_market.py data/training data/rounds/final-round --paths 30 --bars 252
```

The output directory must not already exist. A fresh 128-bit seed is generated
when `--seed` is omitted. For a separate public practice dataset, use a different
output directory and an explicitly public seed:

```powershell
python synthetic_market.py data/training data/rounds/practice --seed 12345
```

The final bundle contains:

- `prices.npz`: all generated prices, including future bars; keep private.
- `manifest.json`: generator version, dependency versions, configuration, asset
  universe, training/model/price hashes, and a seed commitment; no raw seed.
- `private_seed.json`: the final seed; keep private and never distribute to teams.

Keep the entire final directory out of participant download routes and worker
filesystems. Its path is ignored by Git under `data/`. The generation command
prints only a dataset commitment, not the private seed. Publishing this
commitment before submissions close allows organizers to show later that the
dataset was fixed in advance. After the round, reveal the bundle and seed if
the club wants independent reproduction.

## Reuse the frozen prices

```python
from synthetic_market import SyntheticRound

round_data = SyntheticRound.load("data/rounds/final-round")
prices = round_data.path(0, ["SPY", "QQQ"])
# prices["SPY"] has integer steps -100 through 251.
# Only the current history prefix may cross into a strategy worker.
```

Loading verifies the price hash and market invariants. The stored array is
read-only; each call to `path` returns independent DataFrames. Use this saved
bundle for every submission and re-score, rather than downloading market data
or regenerating the round. A future evaluation ID must combine its `dataset_id`
with the execution engine version and all scoring/execution settings.
Pass the recorded commitment as `expected_dataset_id` to `SyntheticRound.load`
when verifying that both the manifest and the prices match the original round.

Regeneration with the same training data, seed, configuration, and dependency
versions is deterministic. Frozen snapshots remain the source of truth across
dependency upgrades, which can change numerical or random implementations.

## Verify offline

```powershell
python -B -m unittest discover -s tests -v
```

Checks cover fresh warm-ups/candles/volumes, independent streams, ticker-order
fairness, cross-asset correlation, changing volatility, seed/config validation,
strict input calendars, snapshot integrity, and reproducible reloads without
Yahoo Finance or network access.
