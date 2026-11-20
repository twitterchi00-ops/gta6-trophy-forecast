[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/twitterchi00-ops/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/twitterchi00-ops/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy earn-rate forecaster — make.fun `gta6-trophy-rarity-forecast-01`

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01** ("Forecast how
many GTA VI players earn each trophy in week one", $250 USDC top-20, free
entry, deadline 21 Nov 2026 15:00 Tehran). `main.py` (Python 3.13, stdlib
only) reads a case file and forecasts, for each of the 55 PS5 trophies, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case's target.
Score = mean pinball loss on the logit scale.

## Why the house baseline loses

An Exophase earn rate is a ratio: `rate = completions / players_tracked`.
Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously. On launch night the page sat unchanged (638 players tracked,
every rate identical) from 23:48 UTC through the 04:00 UTC read, then
refreshed by 04:30 UTC: 3,868 players tracked and every rate but the 100%
one LOWER. The house baseline (last rate held, narrow random-walk spread)
scores ~1.51 on the practice because the refresh dilutes every rate by
roughly 5x and its band is far too narrow to contain it (coverage 0).

## The model

This forecaster models the ratio's two parts separately:

**N(target) — players tracked at the target.** When the history shows N
moving, a power-law fit of `ln N` on `ln(1 + hours since release)` (whole
history blended with the last 24 h, 2:1 in log space, capped at 30x the last
read) is extrapolated. When the history is FLAT (the practice cases: N
stuck at 638), no trend exists to fit, so a documented prior fires:
Exophase's first refresh multiplied tracked players by 6.06x (638 → 3,868,
stated in RULES.txt), so `N(target) = 6.06 * N(last)`.

**C_i(target) — completions of trophy i.**

    C_i(target) = C_i(last) + rho_i * (N(target) - N(last)) + lambda_i * hours

`rho_i` is the fraction of newly tracked players who earn trophy i,
`lambda_i` the per-hour earning rate of the players already tracked. They
come from a per-trophy SEQUENTIAL fit: first the slope of completions on
players tracked (rho), then half of the leftover time trend of the residuals
(lambda). A joint fit was tried and rejected: players tracked and time are
~0.98 correlated, so a joint fit splits their shared growth arbitrarily and
inflates rho. Both fitted values are shrunk 50% toward their tier median
and blended with a prior by an identification weight (0 when N never moved,
1 once N has grown 5x+). When the history is flat the fit is unidentified
and a prior is used: `rho_i = 0.05 * r_i(last)` — newly tracked players
earn at a small fraction of the current rate. The 100% trophy stays at 100%
(it did so through a 6x refresh). On the varying path rho gets a tenure
uplift of 0.35 per `ln(1 + horizon_days)`.

**Spread.** Quantiles are the median shifted in logit space by `z_tau *
sigma`, where sigma combines the house random-walk term (0.5·√days), a
robust fit-scatter term (floor 0.3, cap 0.9), a dilution term
(0.30·√days), and — only for flat history — a refresh-uncertainty term
(0.6). Practice sigma ≈ 0.6; final-horizon sigma ≈ 1.53. Sigma is floored
at 0.35 and capped at 2.0.

**Constants:** `F_PRIOR=0.05`, `JUMP_PRIOR=6.06`, `SIGMA_FLAT=0.6`,
`SIGMA_RW=0.5`, `SIGMA_DILUTION=0.30`, `LAMBDA_SHRINK=0.5`,
`RHO_UPLIFT=0.35`.

## Verification (all run in this session)

The kit ships no practice answers, so `work/score.py` scores against
documented proxy answers (see its docstring): the refresh model
`C' = C + f·(r/100)·(3868−638)`, calibrated so the starter reproduces the
kit's published house numbers (loss ~1.51, miss ~0.68, coverage 0) — the
calibration lands at f = 0.04.

**Practice (p00 + p01 pooled, 550 items), entry vs starter:**

| Proxy truth | Entry | Starter |
|---|---|---|
| f = 0.00 pure dilution | 0.164 | 1.687 |
| f = 0.033 | 0.133 | 1.535 |
| **f = 0.04 calibrated (central)** | **0.127** | **1.506** |
| f = 0.066 | 0.132 | 1.404 |
| f = 0.10 | 0.156 | 1.284 |
| f = 0.15 | 0.186 | 1.132 |
| f = 0.25 | 0.310 | 0.883 |
| f = 0.40 | 0.481 | 0.598 |
| β = 1.3 | 0.297 | 0.905 |
| β = 1.5 | 0.151 | 1.308 |
| β = 1.7 | 0.131 | 1.522 |
| β = 2.0 | 0.155 | 1.644 |

The entry beats the starter in **every** scenario (worst case 0.481 vs
0.598). Central estimate: **0.127 vs 1.506 — ~12x better**, typical miss
0.007 pp vs 0.697 pp, coverage 0.982 vs 0.000.

**Synthetic final-horizon case** (`work/synth2.py`, cohort-based truth with
a different mechanism than the entry's model — N 209k → 398k, rates rise
1.3–3.8x): **entry 0.442 / miss 10.61 / cov 1.000** vs starter 0.562 /
16.45 / 0.964.

**Determinism / runtime / format.** Two runs on the practice cases and on
the synthetic case are byte-identical (`cmp` clean). Runtime ~0.19 s for
both practice cases together (limit 8 s per case). Grader-style run
(`PYTHONHASHSEED=0 python3 -P -s main.py`, only `main.py` present)
reproduces the same bytes. Every output validated: exactly
`{"trophies": [...]}`, 55 rows, each id once, each `q` five finite numbers
0–100, non-decreasing, ≤ 65,536 bytes (practice outputs: 2,802 bytes).
Edge cases (single-read history, empty history, null rates, a 0.00% rate,
a never-observed trophy) all produce valid output.

## Honest caveats

1. **The 0.127 is a proxy estimate, not a measured score.** The true
   practice answers are hidden on the server. The honest band is the
   scenario range 0.127–0.481.
2. **Practice horizon (90 min) ≠ final horizon (4.5 days).** The
   flat-history prior that drives the practice win never fires on the real
   final; the final path is validated only on synthetics.
3. **The house AI-model bar is invisible**, and so are other entrants'
   forecasts. Beating 1.51 on the practice is necessary, not sufficient:
   the prize needs beating the best house entry *on the final* and ranking
   top 20.

## How to run

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every `*.in` in the input dir, writes `<id>.out` per case. No
network, no randomness, no clock reads — every time used comes from the
case file.

## Files

- `main.py` — the entry (Python 3.13, stdlib only)
- `work/score.py` — practice proxy scorer (pinball loss per RULES.txt)
- `work/synth2.py` — synthetic final-horizon case generator
- `system-prompt.txt` — the task instructions this build was made under
- `README.md`, `LICENSE` — this file, MIT license
