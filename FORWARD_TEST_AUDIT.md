# Forward-test infrastructure audit

Reviewed on 7 November 2026. The starting revision was `ca59fd5` on
`fix/competition-evaluation`. Tests used the existing DeepSeek strategies,
the supplied six public datasets, the real authenticated native worker under
unprivileged Ubuntu/WSL, the competition database workflow, and Streamlit's UI
test runner.

## Findings and changes

### Worker failures could become published team failures

Before the fix, occupying both worker slots made final scoring record
`StrategyError: Worker is busy. Try again shortly.` against a valid selection.
The round became `scored`, and publication succeeded. Practice could occupy
both slots while another round was being evaluated.

Worker capacity, authentication, transport, OS-launch, and runtime failures now
use retryable infrastructure exceptions. Grading releases its lease, returns
the round to `locked`, preserves completed teams, and leaves the interrupted
selection pending. Genuine strategy failures remain recorded for the organizer.
Unexpected controller exceptions also preserve pending work. Publication remains
blocked until grading has completed.

The default two-session worker reserves one slot for finals. Online practice
closes when submissions lock. The native HTTP rehearsal exercised practice
contention, deliberate final-session saturation, blocked publication, release
of capacity, successful retry, and publication.

### Evaluation identity omitted supporting controller code

Changing a supporting module previously left `evaluation_id` unchanged. The
engine hash covered `backtester.py`, while preprocessing and protocol behavior
also depended on other modules.

The identity now includes hashes of `market_data.py`, `synthetic_market.py`,
`secure_random.py`, and `strategy_runner.py`. The existing dataset, engine,
configuration, numerical-library, and worker-runtime identities also participate.
Regression tests change each supporting file independently and verify invalidation.

### Worker runtime was checked only before the grading job

A worker restart between paths could introduce a different runner or library
version after the initial runtime check. The worker also advertised its startup
fingerprint even if its files were replaced before a restart.

Every final-path session now requests the round's pinned runtime and verifies
the response. The worker checks current files and installed package versions
before creating a session and when queried for its runtime. The aggregate is
returned only after a final runtime check. A mismatch leaves grading pending.
Tests cover changed fingerprints and the per-path request contract. A real
native HTTP request with an incorrect fingerprint was rejected before launch.

### The member kit exposed the exact simulator implementation

The original kit included `synthetic_market.py`, including calibration formulas
and simulator constants. DeepSeek used it for its synthetic validation. Public
data loading depended on that module, which pulled organizer preparation into
the participant workspace.

Shared schemas and CSV validation now live in `market_data.py`. The revised
member kit includes the execution engine, loader, example, data, settings,
dependencies, and README. Generator and private-randomness modules stay in the
organizer application.

The repository still contains the simulator source. Anyone with repository
access can inspect the formulas. Excluding it from the kit separates the
participant workflow from round preparation. Final-path privacy depends on
secret draws and isolation. Participants can learn the model's relationships
from their datasets and any source material they can access.

### Private market randomness needed additional hardening

The earlier simulator used NumPy's default statistical PRNG. NumPy documents
that its generators are unsuitable for security-sensitive purposes. Our audit
did not demonstrate practical seed recovery from observed candles.
[NumPy documentation](https://numpy.org/doc/stable/reference/random/index.html)

Generator v3 uses a reproducible HMAC-SHA256 counter stream derived from a fresh
128-bit round seed. Separate labels scope path, candle, and volume draws.
The manifest records the randomness version. Known-answer, chunking, stream
separation, bounds, and distribution sanity tests cover the implementation.
The implementation uses Python's standard HMAC primitive.
[Python HMAC documentation](https://docs.python.org/3/library/hmac.html)

### Native isolation now has a startup check

The worker launches a trusted probe through the actual sandbox before binding
its listener. The probe verifies host-file isolation, read-only storage,
blocked sockets, cleared worker credentials, separate mount/process/network
namespaces, and memory/CPU limits. Final round preparation requires the worker's
verified status. A failed policy check prevents service startup.

## Privacy and execution checks

The real worker passed the following checks.

- Modifying future OHLCV and interest-rate observations at step 11 left every
  earlier strategy observation identical. The first changed digest appeared at
  step 11.
- Guest `__main__`, caller-frame inspection, environment reads, direct host-file
  access, and `/proc/1/root` access did not expose the evaluator's canaries or
  private variables.
- Native tests blocked sockets, writes, forks, oversized memory mappings,
  output flooding, and runaway execution. NumPy/pandas strategies still ran.
- Blind-round UI tests showed zero final metrics or equity charts before
  publication, including for a signed-in team. Publication rendered the results
  only after selections were locked and grading completed.
- Synthetic tests verified fresh warm-ups, candles and volumes, independent
  path streams, prefix preservation, ticker-selection fairness, snapshot hashes,
  macro histories, and reloads of frozen data.
- All three existing DeepSeek strategies completed the 4,680-step public
  practice sample through the real worker. Their historical scores matched the
  earlier review.

The suite contains 91 cases. Windows completed its applicable cases with ten
Linux-native skips. Unprivileged Ubuntu completed its applicable cases with
five skips covering the Streamlit environment and missing-sandbox check. Ruff
passed for the changed Python files. The full suite is run with
`python -B -m unittest discover -s tests -q`.

## Calibration stress results

Each column uses 30 independent one-year paths fitted to its stated public
period. Diagnostic seed 3407 is shared across calibrations. Costs, warm-up,
strategy source, and execution conventions are identical. These runs are
research diagnostics and may be inspected publicly.

| DeepSeek strategy | Full public 2007–2025 | Earlier 2007–2016 | Recent 2017–2025 |
| --- | ---: | ---: | ---: |
| Equal weight | 0.352 | 0.364 | 0.521 |
| Momentum | -0.268 | -0.331 | 0.215 |
| Rate-aware risk parity | 0.296 | 0.226 | 0.519 |

Values are median Sharpe. For the full-public calibration, the conditional 95%
bootstrap intervals are approximately `[-0.004, 0.688]` for equal weight,
`[-0.806, 0.422]` for momentum, and `[0.044, 0.724]` for rate-aware risk parity.
The close equal-weight/rate-aware results do not establish a reliable winner.
The earlier-versus-recent momentum results show sensitivity to calibration.

The official generator still uses a single full-public-data calibration.
The stress panel checks sensitivity without changing that selection rule.
Learning its distribution remains possible. The model represents aggregate
covariances, bounded serial dependence, volatility regimes, and a simplified
rate process. It cannot establish real-market performance or reproduce every
economic mechanism. Confidence intervals describe Monte Carlo uncertainty
conditional on that model.

Before freezing competition rules, choose the path count and any mixture of
calibration regimes in advance. Compare candidates over additional diagnostic
draws. Record those decisions before generating the final round and accepting
entries, so strategy outcomes cannot influence the chosen final distribution.
The existing 4% Sharpe reference, zero cash interest, 100-bar warm-up, and
long-only execution rules also need explicit approval as final rules.

## Artifacts and deployment scope

The revised member deliverable is
`data/deliverables/nova-quant-member-starter-v3.zip`. The earlier member workspace
and ZIP are retained for reproducing DeepSeek's work. Runtime evidence is saved
locally under `data/reviews/robustness/`, including the before/after results and
the audit harness. These ignored files contain rehearsal data.

Saved v1/v2 market bundles remain readable. New rounds use generator v3 and
engine `streamed-next-open-v5`. Finishing an older locked round requires its
pinned controller and worker versions. New code intentionally produces a new
evaluation identity.

Verification covered the local Windows/Ubuntu environments. The live
Coolify host, native-worker service, TLS route, and deployment resource limits
still require operator verification. The 45-second session lifetime includes
transport time, so full public-history practice must also be profiled through
the deployed route. The native sandbox shares the host kernel. Current patching
and the dedicated runtime layout in [DEPLOYMENT.md](DEPLOYMENT.md) are part of
its operating requirements.
