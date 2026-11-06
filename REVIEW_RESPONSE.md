# Response to the jitter review

The security, data-identification, adaptive-submission, and reproducibility
concerns were valid. This branch replaces the historical final-test approach
rather than trying to hide real SPY returns with larger perturbations.

| Finding | Resolution |
| --- | --- |
| Strategies inspect evaluator frames, secrets, or database | Every web upload import, constructor, and run call executes in a separate native Linux sandbox service. Only current history crosses the JSON protocol. No host data/secrets are mounted or inherited. |
| Unmodified warm-up, volume, and candle ratios fingerprint history | Generate new warm-up bars and OHLCV values. No hidden historical sequence is replayed. |
| Thirty perturbations share the same real path | Independent random streams produce distinct synthetic paths with correlated assets and changing regimes/volatility. |
| Benchmark and strategy output reveal the holdout | No hidden test runs while submissions are open. Metrics, errors, and curves are published only after final selections lock and evaluations finish. |
| Repeated feedback supports hill-climbing | Practice uses only public data. Team-scoped version limits/cooldowns and a frozen final selection prevent name changes from resetting limits. |
| Members can impersonate other members | Team identity is authenticated with separate random access tokens; editable author/name fields do not identify the leaderboard owner. |
| Re-scoring may download revised market data | Freeze public and private datasets, hash prices/metadata, and pin generator/engine/settings/library/runtime identities. Re-scoring never downloads data. |
| First-day benchmark advantage | Both portfolios begin at the first close in cash; both first fill opportunities are the next open, with the same costs. |
| Temp-file leaks and sanitized-name collisions | Web submissions are immutable source versions in SQLite, keyed by UUID and content hash. Sandbox temporary source files are cleaned on exit/failure. |
| Invalid seeds and incomplete cache identity | Validate 128-bit non-negative seeds; use no public final seed default. Evaluation identity includes the dataset, all engine settings, and numerical/worker runtime versions. No benchmark cache is used. |

Two qualifications to the original review:

- A plain subprocess with an empty environment would not have been sufficient:
  it could still read the app's files, communicate over the network, and access
  other processes. The worker adds kernel-enforced namespace/filesystem/seccomp
  restrictions and resource limits, with no unsandboxed fallback.
- The review's exact correlation and Sharpe uncertainty figures depend on its
  fixture and statistical assumptions. They correctly illustrate the weakness
  of small jitter, but are not universal error bounds for leaderboard gaps.

Final ranking uses median Sharpe, path P10/P90, and a bootstrap interval for the
median under the fixed generator. These do not measure real-world model error.
Synthetic evaluation is an explicit model-based competition; calibrate and
inspect it on the actual public dataset before running the event.

The real sandbox was exercised locally under Ubuntu, including native NumPy /
pandas imports, forbidden host files/secrets/network/writes/fork, resource
limits, and an end-to-end authenticated worker evaluation. The live Coolify
deployment still needs operator setup and verification described in DEPLOYMENT.md.
