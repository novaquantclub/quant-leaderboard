# DeepSeek member-package review

Reviewed on 7 November 2026 using the standalone member workspace, the native
Linux worker under WSL, and the competition application's normal submission and
scoring entry points.

## Package comprehension

DeepSeek successfully implemented all three requested strategies and provided
`strategy.py` as a self-contained equal-weight submission. Its `Trader` class
is identical to the class in `equal_weight.py`. All 15 original package files,
including the six CSVs, engine settings, README, and backtesting modules, match
the original ZIP byte for byte.

The strategies declare valid metadata, use normalized prices and simulation
units, submit position changes for next-open execution, and access rates through
`state.observations["interest rate"]`. Their runtime dependencies are NumPy and
pandas. All submitted files imported and executed inside the native sandbox.

DeepSeek also supplied chronological validation, a rate-blind comparison, turnover
measurement from filled position changes, and a detailed report of its choices.
The report identifies the reuse of validation results during some design decisions.

## Reproduced results

All 16 historical comparisons across the three validation folds and the continuous
2017–2025 period reproduced through the authenticated HTTP worker. The largest
absolute difference across the checked Sharpe, return, drawdown, and turnover
metrics was below `9e-14`.

The four 30-path synthetic comparisons at public seed 12345 also reproduced.
The largest median-Sharpe difference was below `6e-15`.

The fresh rehearsal used 60 newly generated one-year paths, shared across teams.
Main comparisons use 100 warm-up bars, 10 bps trading costs, a 4% annual Sharpe
reference, zero cash interest, and an initial balance of 100,000.

| Strategy | Historical 2017–2025 Sharpe | Reproduced 30-path median | Fresh 60-path median |
| --- | ---: | ---: | ---: |
| Equal weight / preferred submission | 0.668 | 0.611 | 0.705 |
| Momentum | 0.365 | 0.243 | 0.375 |
| Rate-aware risk parity | 0.585 | 0.536 | 0.771 |
| Rate-blind risk-parity control | 0.555 | 0.496 | 0.767 |

The ATLAS comparison benchmark scored 0.768 on the historical validation and
0.739 median Sharpe on the fresh synthetic paths.

The leading strategy changed across the diagnostic samples. On the fresh paths,
rate-aware exceeded equal weight by 0.066, with a 95% paired bootstrap interval
for that advantage of approximately `[-0.185, 0.399]`. Its advantage over the
rate-blind control was 0.004, with interval `[-0.198, 0.229]`. Equal weight's
advantage over momentum was 0.330, with interval `[0.105, 0.707]`.

These intervals describe sampling uncertainty under the fitted generator.
The rate feature's performance contribution remains uncertain for this strategy.

## Infrastructure failure and fix

The original transport resent the complete growing history at every step. A
full-history equal-weight run exceeded the 45-second limit while attempting
step 1,567 of 4,680 scored steps.

The worker now negotiates incremental history updates. It receives the initial
history prefix followed by newly available rows. The guest reconstructs the full
history for each call using independent copies. Historical revisions trigger a
full replacement. Older workers receive full states through the compatibility
path. The resource and isolation limits continue to bound each sandbox session.

With the final transport revision, all three strategies completed the full
4,780-row public dataset through the native HTTP worker.

| Strategy | Full-history hosted runtime |
| --- | ---: |
| Equal weight | 15.08 s |
| Momentum | 13.65 s |
| Rate-aware | 15.36 s |

The final revision also matched all four strategies' results on the first frozen
path of the fresh rehearsal. Regression tests cover transmitted row counts,
historical revisions, legacy workers, repeated steps, asset ordering, malformed
updates, and mutation isolation.

## Competition workflow

The rehearsal used a separate database and four authenticated teams. It passed
14 checks covering duplicate-upload idempotency, cooldowns, token rotation,
cross-team selection rejection, preferred-version selection, submission limits,
blind scores, open-round scoring rejection, locked uploads and selections,
repeat scoring, publication, and published-round rescore rejection.

All four selected entries scored successfully. Worker sessions were cleaned up.
UI checks used copies of the rehearsal database. Private results produced zero
metric tables and zero charts. Publication rendered all four teams in score order
and displayed the comparison chart. An automated UI regression test covers this
publication boundary.

## Research issues to address

- The rate-aware construction and rebalance interval were checked against the
  validation folds during development. Those validation results include model
  selection. DeepSeek discloses this in its report.
- `research_rate.py` computes forward 21-day volatility before slicing to the
  research end date. The final 21 research labels include validation observations,
  reaching 1 February 2017. Purge labels whose observation horizon extends beyond
  the research boundary when rebuilding that table.
- The risk-parity solver exits early on a non-positive risk contribution. In
  108 full-validation rebalances, two target allocations had negative risk
  contributions. The median maximum deviation from equal risk budgets was about
  `7e-11`, so the method achieved its stated target on most rebalances. The report
  should identify its approximate-solver fallback.
- The rate strategy requests a 252-return covariance lookback while the pilot
  supplies 100 warm-up bars. Its early estimates use the available shorter sample.
  A 252-bar warm-up would support a full one-year lookback at the first scored bar.

## Decisions for the rules discussion

Choose the supported lookback and warm-up together. Assess path count and horizon
using repeated public diagnostic rounds, with an explicit policy for close scores.
Choose the benchmark portfolio and specify the relationship between cash accrual,
the Sharpe reference rate, and the observed macro series. The current results
support further checks of those choices before freezing a competition round.

## Evidence and deployment scope

Machine-readable results, input hashes, workflow checks, scripts, and UI results
are stored under `data/reviews/deepseek/`. The fresh diagnostic dataset was created
through `Competition.create_round` and scored through `Competition.score`.
The source strategies in the separate member workspace retain their original
contents.

Native execution was tested locally on Linux/WSL over authenticated loopback HTTP.
The actual Coolify host requires its own isolation, resource, and HTTPS latency
checks during deployment.
