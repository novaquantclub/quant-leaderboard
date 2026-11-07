# Nova Quant League

A team strategy competition with a public training dataset, an online practice
backtester, immutable strategy versions, and blind final testing on independent
synthetic markets.

## Competition flow

1. An organizer prepares the public OHLCV CSVs and opens a round. The app freezes
   the public data, generates the private final dataset once, and records a
   dataset commitment and evaluation identity.
2. Organizers issue a separate random access token to each team.
3. Teams download the same public dataset, practice locally or online, and
   upload strategy versions. Practice never uses the final dataset.
4. Each team explicitly selects one final version. Upload limits and cooldowns
   apply to the authenticated team, independently of the strategy's name or author.
5. The organizer locks submissions and selections, then evaluates those frozen
   versions. Final metrics, errors, benchmark values, and curves remain private.
6. The organizer publishes the finished leaderboard. Ranking uses median Sharpe
   across independent synthetic paths, with path dispersion and a conditional
   bootstrap interval shown alongside it.

## Strategy API

Start with `sample_strategy.py`:

```python
class Trader:
    name = "My strategy"
    author = "Your name"  # Descriptive; the login token determines the team.
    tickers = ["ATLAS", "BOREAL", "CIRRUS", "DUNE", "EMBER"]

    def run(self, state):
        return {"ATLAS": 10} if state.step == 0 else {}
```

`run` receives `step`, `history`, `positions`, `cash`, `portfolio_value`, and
`trader_data`. History contains OHLCV bars from `-100` through the current step.
Return an order dictionary or `(orders, trader_data)`. Persistent string data
must be no larger than 100 KB; uploaded Python files are limited to 128 KB.

Strategies should be deterministic. Standard Python/NumPy RNGs use a public seed
of zero in each fresh worker; explicitly seed independent generators such as
`numpy.random.default_rng` and model `random_state` parameters as well. Wall-clock
time and native entropy are not virtualized, so arbitrary randomized code is
not guaranteed to reproduce just because the market snapshot is frozen.

Orders fill at the next open with 10 bps costs. Positions are long-only; symbols
containing `^` or `=` are signals and cannot be traded. Both portfolios start in
cash at the first close. The selected benchmark asset buys at the following open,
which is the first fill opportunity available to a strategy.

## Public data and local practice

Install Python 3.10+ and the application requirements in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Teams receive five neutrally named assets, ATLAS, BOREAL, CIRRUS, DUNE, and EMBER.
Researching their behaviour and discovering relationships is part of the challenge.
The public training period covers 2007 through 2025 with daily observations.

Organizers can supply prepared CSVs or download them once. In PowerShell,
`$sourceTickers` is the organizer's private five-symbol list.

```powershell
python prepare_training.py data/training --start 2007-01-01 --end 2026-01-01 --tickers $sourceTickers --asset-names ATLAS BOREAL CIRRUS DUNE EMBER
```

The command randomly assigns the competition names and records the source mapping,
download settings, and CSV hashes in the sibling `data/training.private.json`.
Keep that organizer-only file separate from participant downloads. Files ending
in `.private.json` are ignored by Git. Historical patterns can still identify a
source asset, so the blind final uses newly generated paths.
The exporter repairs adjusted-price ordering differences within eight float64
rounding steps. Larger OHLC ordering errors fail validation.

Each `SYMBOL.csv` needs an ordered Date/Step index and
`Open,High,Low,Close,Volume`. Assets must share a complete calendar and have at
least 120 bars, including the 100-bar warm-up. Downloads use the shared calendar
intersection without backfilling. Never revise a published dataset in place.

Teams can download the training ZIP and SDK from the app, then run their own
code locally without an online worker:

```powershell
python backtester.py sample_strategy.py --data training --benchmark ATLAS --trusted-local
```

The organizer selects a benchmark by its competition name when opening a round.
The downloaded starter uses that round's asset names, and the SDK and dataset
README explain the schema and execution settings. Teams investigate the asset
relationships themselves.

`--trusted-local` executes the named script on the user's own machine. Use it
only for code you trust. The web application has no corresponding fallback.

## Hosting and isolation

The web app stays on **Coolify with Nixpacks**, using the existing start command:

```text
streamlit run app.py
```

Hosted uploads run in a separate **native Linux worker**, using bubblewrap and
seccomp. No Docker daemon, Dockerfile, privileged web container, or Docker socket
is required for the strategy runner. See [DEPLOYMENT.md](DEPLOYMENT.md) for the
native service and Coolify connection setup.

Every import, constructor, and `run` call executes inside that worker. It gets
only the source and current trading state. The app's database, passwords,
private dataset, seed, and future bars are never sent to it or mounted into the
guest. The guest has a read-only filesystem, separate process/network/user
namespaces, no network sockets, and CPU, memory, process, output, and time limits.
Unsupported or unavailable isolation causes evaluation to fail closed.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `ADMIN_PASSWORD` | Organizer password, at least 32 characters; no usable default |
| `NQC_WORKER_URL` | Authenticated worker endpoint; HTTPS except localhost |
| `NQC_WORKER_TOKEN` | Separate random worker credential, at least 32 characters |
| `DB_PATH` | SQLite path, default `data/leaderboard.db` |
| `ROUND_ROOT` | Frozen round storage, default `data/rounds` |

Set these through Coolify's environment configuration; the app does not
automatically load `.env`. Keep `data/` on persistent storage. Team tokens are
generated in the organizer panel and stored only as hashes. Club passwords,
historical source windows, and public default OOS seeds are no longer used.

Existing legacy tables and uploaded files are left in place. They do not enter
the new rankings. Start a new round, issue team tokens, and have teams upload
and select their final versions; legacy uploads are not automatically executed.

## What synthetic evaluation means

Final paths are newly simulated OHLCV histories, including warm-up and volume,
calibrated from the published data. They use joint overnight/intraday covariance,
bounded return autocorrelation, heavy-tailed shocks, changing volatility, and
calm/stress regimes. There is no real historical final path to identify.

Each round saves the generated prices and their hash. Every team uses the same
universe, paths, execution settings, and worker runtime. Dataset, code, settings,
and dependency changes cannot silently reuse an existing evaluation identity.

This measures performance under the declared model. It does not establish
future real-market performance. The displayed interval describes Monte Carlo
uncertainty conditional on the model; it does not include model error. Inspect
the model's realism and baseline strategies on the actual public dataset before
running a competition. See [SYNTHETIC_ROUNDS.md](SYNTHETIC_ROUNDS.md) and
[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md).

## Verification

```powershell
python -B -m unittest discover -s tests -v
```

Run the same suite under an unprivileged Linux account with bubblewrap and
libseccomp installed to exercise the real sandbox. Windows checks the frontend,
engine, database, snapshots, protocol, and refusal to run without isolation.
Linux additionally tests native-library strategies, host-file/secret/network
isolation, fork prevention, read-only files, memory/output/time limits, and an
end-to-end HTTP worker evaluation on frozen synthetic data.
