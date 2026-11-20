"""gta6-trophy-rarity-forecast-01 entry: dilution-aware trophy earn-rate forecaster.

Python 3.13, standard library only. Reads every <id>.in in the input directory
($MAKEFUN_INPUT_DIR or /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR
(or /output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.

WHY THE HOUSE BASELINE LOSES
----------------------------
An Exophase earn rate is a ratio: rate = completions / players_tracked.
Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously. On launch night the page sat unchanged (638 players tracked,
every rate identical) from 23:48 UTC through the 04:00 UTC read, then
refreshed by 04:30 UTC: 3,868 players tracked and every rate but the 100%
one LOWER. The house baseline (last rate held, narrow random-walk spread)
scores ~1.51 on the practice because the refresh dilutes every rate by
roughly 5x and its band is far too narrow to contain it (coverage 0).

MODEL -- forecast the ratio's two parts separately
-------------------------------------------------
N(target) -- players tracked at the target:
  * When the history shows N moving, extrapolate a power law: ln N against
    ln(1 + hours since release), fit on the whole history and again on the
    last 24 h of reads, blended 2:1 (log space) in favour of the full
    history, floored at the last read and capped at 30x it.
  * When the history is FLAT (the practice cases: N stuck at 638), no trend
    exists to fit, so a documented prior fires: Exophase's first refresh
    multiplied tracked players by 6.06x (638 -> 3,868, stated in RULES.txt),
    and a refresh is expected inside the horizon after hours of staleness,
    so N(target) = 6.06 * N(last).

C_i(target) -- completions of trophy i:
    C_i(target) = C_i(last) + rho_i * (N(target) - N(last))
                             + lambda_i * (hours to target)
  rho_i is the fraction of newly tracked players who earn trophy i,
  lambda_i the per-hour earning rate of the players already tracked.
  They come from a per-trophy SEQUENTIAL fit over the history: first the
  slope of completions on players tracked (rho), then half of the leftover
  time trend of the residuals (lambda). A joint fit was tried and rejected:
  players tracked and time are ~0.98 correlated, so a joint fit splits
  their shared growth arbitrarily and inflates rho. Both fitted values are
  shrunk 50% toward their tier median (platinum / gold / silver / bronze
  trophies behave alike) and clipped to sane ranges. The fitted values are
  blended with a prior by an identification weight that is 0 when N never
  moved and 1 once N has grown 5x or more.
  When the history is flat the fit is unidentified and a prior is used:
  rho_i = F_PRIOR * r_i(last) -- newly tracked players earn at a small
  fraction of the current rate (most of them have barely played; the
  published house practice numbers -- loss ~1.51, typical miss ~0.68,
  coverage 0 -- are jointly reproduced by F ~ 0.04, and the shipped
  F_PRIOR = 0.05 sits at the centre of the consistent band).
  The 100% trophy is the exception that proves the rule: it stayed at 100%
  through a 6x refresh, so its rho is 1 and it stays at 100%.
  On the varying path rho gets a tenure uplift: the fitted rho reflects
  joiners with short tenure; joiners by the target have up to a full
  horizon longer to earn.

Median rate m_i = 100 * C_i(target) / N(target), completions floored at
C_i(last) (they never decrease) and capped at N(target).

SPREAD -- honest about a refresh the history has not seen
--------------------------------------------------------
Quantiles are m_i shifted in logit space by z_tau * sigma:
  * a random-walk term 0.5 * sqrt(horizon_days) (the house term);
  * a fit term from the scatter of the history's logit rates (floor 0.3,
    cap 0.9, robust median-absolute-deviation scale);
  * a dilution term 0.30 * sqrt(horizon_days) for N / rho uncertainty;
  * when the history is flat, a flat-history term (SIGMA_FLAT) for the
    refresh itself: whether one lands before the target, and how big.
Sigma is floored at 0.35 and capped at 2.0 logit units. The final's
4.5-day horizon therefore gets a much wider band than the practice's
90 minutes, as it must: one refresh over 90 minutes is little guide to
a week, and the house baseline's final row for a 1.10% trophy already
spans 0.19% to 6.01%.

Deterministic: no randomness, no network, no clock reads (every time used
comes from the case file), files processed in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuned constants (grid-chosen; see README.md) ---
F_PRIOR = 0.05          # flat-history prior: new players earn at 5% of the current rate
JUMP_PRIOR = 6.06       # flat-history prior: first refresh multiplied players 638 -> 3868
SIGMA_FLAT = 0.6        # flat-history spread: the refresh itself is uncertain
SIGMA_RW = 0.5          # random-walk logit sigma per sqrt(day), as the house baseline
SIGMA_DILUTION = 0.30   # dilution-uncertainty sigma per sqrt(day)
SIGMA_FIT_FLOOR = 0.30
SIGMA_FIT_CAP = 0.90
LAMBDA_SHRINK = 0.5     # fraction of the leftover time trend kept as lambda
RHO_UPLIFT = 0.35       # rho tenure uplift per ln(1 + horizon_days), varying path only
SIGMA_MIN = 0.35
SIGMA_MAX = 2.0
N_GROWTH_CAP = 30.0    # never forecast N beyond 30x the last read


def parse_time(text):
    """Parse an ISO 8601 timestamp (handles the trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    lg = max(-700.0, min(700.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (same convention as the starter)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def ols(xs, ys):
    """Least squares ys ~ a + b*xs. Returns (intercept, slope)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (ys[0], 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - slope * mx, slope)


def load_reads(case):
    """Normalise the case history into (hours-since-release, N, rates) rows."""
    try:
        t0 = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    except Exception:
        t0 = None
    rows = []
    for h in case.get("history") or []:
        try:
            at = parse_time(h["at"])
        except Exception:
            continue
        n = h.get("players_tracked")
        rates = h.get("rates") or []
        if n is None or not rates:
            continue
        if t0 is None:
            t0 = at
        hours = (at - t0).total_seconds() / 3600.0
        rows.append((hours, float(n), list(rates)))
    rows.sort(key=lambda r: r[0])
    return rows


def forecast_n(reads, horizon_hours):
    """Forecast players tracked at the target. Returns (n_target, flat)."""
    n_last = reads[-1][1]
    n_min = min(r[1] for r in reads)
    n_max = max(r[1] for r in reads)
    flat = (n_max <= n_min * 1.000001
            and all(r[2] == reads[0][2] for r in reads))
    if flat:
        return n_last * JUMP_PRIOR, True

    t_target = reads[-1][0] + horizon_hours
    x_target = math.log1p(max(0.0, t_target))

    def power_law_fit(subset):
        xs = [math.log1p(max(0.0, r[0])) for r in subset]
        ys = [math.log(max(1.0, r[1])) for r in subset]
        a, b = ols(xs, ys)
        return math.exp(min(50.0, a + b * x_target))

    n_all = power_law_fit(reads)
    recent = [r for r in reads if r[0] >= reads[-1][0] - 24.0]
    if len(recent) >= 5 and len({r[1] for r in recent}) >= 3:
        # Weight the full history 2:1 over the recent window: recent
        # log-log slopes are noisy and extrapolate badly when growth is
        # saturating (in tests the recent-only fit overshot 2x).
        n_recent = power_law_fit(recent)
        n_target = math.exp((2.0 * math.log(max(1.0, n_all))
                             + math.log(max(1.0, n_recent))) / 3.0)
    else:
        n_target = max(1.0, n_all)
    n_target = min(max(n_target, n_last), n_last * N_GROWTH_CAP)
    return n_target, False


def fit_rho_lambda(reads, trophies, last_rate, flat, horizon_days):
    """Per-trophy (rho, lambda). See the module docstring for the method."""
    nt = len(trophies)
    n_min = min(r[1] for r in reads)
    n_max = max(r[1] for r in reads)
    # Identification strength: 0 when N never moved, 1 once N grew 5x+.
    ident = 0.0 if flat else min(1.0, max(0.0, (n_max / max(1.0, n_min) - 1.0) / 4.0))

    by_tier = {}
    for i, t in enumerate(trophies):
        by_tier.setdefault(t.get("tier", "bronze"), []).append(i)

    # Sequential OLS fits, per trophy: completions on N (rho), then half
    # the leftover time trend of the residuals (lambda).
    rho_fit = [None] * nt
    lam_fit = [None] * nt
    if ident > 0.0:
        ns = [r[1] for r in reads]
        ts = [r[0] for r in reads]
        if len(set(ns)) >= 3:
            for i in range(nt):
                comps = []
                complete = True
                for r in reads:
                    if i >= len(r[2]) or r[2][i] is None:
                        complete = False
                        break
                    comps.append(float(r[2][i]) * r[1] / 100.0)
                if not complete:
                    continue
                a_n, slope_n = ols(ns, comps)
                rho_fit[i] = slope_n
                resid = [c - (a_n + slope_n * n) for c, n in zip(comps, ns)]
                _, slope_t = ols(ts, resid)
                lam_fit[i] = LAMBDA_SHRINK * slope_t

    def tier_shrunk(fitted, idxs):
        vals = sorted(v for v in (fitted[i] for i in idxs) if v is not None)
        return median(vals)

    rho, lam = [], []
    for i, t in enumerate(trophies):
        idxs = by_tier.get(t.get("tier", "bronze"), [])
        if last_rate[i] >= 99.99:
            # The 100% trophy stayed at 100% through a 6x refresh.
            rho.append(1.0)
            lam.append(0.0)
            continue
        # Prior: new players earn at F_PRIOR times the current rate.
        rho_prior = F_PRIOR * last_rate[i] / 100.0
        if rho_fit[i] is None:
            r_i = rho_prior
        else:
            tm = tier_shrunk(rho_fit, idxs)
            fitted = rho_fit[i] if tm is None else 0.5 * rho_fit[i] + 0.5 * tm
            r_i = ident * fitted + (1.0 - ident) * rho_prior
        r_i = min(0.98, max(0.0, r_i))
        if ident > 0.0:
            # Tenure uplift (varying path only): the fitted rho reflects
            # joiners with short tenure; joiners by the target have up to
            # a full horizon longer to earn.
            r_i = min(0.98, r_i * (1.0 + RHO_UPLIFT * math.log1p(horizon_days)))
        rho.append(r_i)
        if lam_fit[i] is None:
            lam.append(0.0)
        else:
            tm = tier_shrunk(lam_fit, idxs)
            fitted = lam_fit[i] if tm is None else 0.5 * lam_fit[i] + 0.5 * tm
            lam.append(max(0.0, ident * fitted))
    return rho, lam


def fit_sigma(reads, nt, horizon_days, flat):
    """Combined logit spread sigma for the forecast quantiles."""
    if flat:
        return SIGMA_FLAT
    # Robust scatter of the history's logit rates around their per-trophy mean.
    devs = []
    for i in range(nt):
        lgs = [logit(float(r[2][i])) for r in reads
               if i < len(r[2]) and r[2][i] is not None]
        if len(lgs) >= 3:
            m = sum(lgs) / len(lgs)
            devs.extend(abs(v - m) for v in lgs)
    if devs:
        sigma_fit = min(SIGMA_FIT_CAP,
                        max(SIGMA_FIT_FLOOR, median(devs) * 1.4826))
    else:
        sigma_fit = SIGMA_FIT_FLOOR
    return math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                     + sigma_fit ** 2
                     + (SIGMA_DILUTION * math.sqrt(horizon_days)) ** 2)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    reads = load_reads(case)

    if not reads:
        # No reads at all: no information. Flat 1% median, widest band.
        return {"trophies": [
            {"id": t["id"],
             "q": [round4(inv_logit(logit(1.0) + z * SIGMA_MAX)) for z in Z]}
            for t in trophies]}

    n_target, flat = forecast_n(reads, horizon_hours)
    n_last = reads[-1][1]

    # Last non-null rate per trophy, and its implied completion count.
    nt = len(trophies)
    last_rate, c_last = [], []
    for i in range(nt):
        rate = None
        for _, _, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        if rate is None:
            rate = 1.0
        last_rate.append(float(rate))
        c_last.append(float(rate) * n_last / 100.0)

    rho, lam = fit_rho_lambda(reads, trophies, last_rate, flat, horizon_days)
    sigma = min(SIGMA_MAX, max(SIGMA_MIN,
                               fit_sigma(reads, nt, horizon_days, flat)))

    rows = []
    for i, t in enumerate(trophies):
        c_t = (c_last[i]
               + rho[i] * max(0.0, n_target - n_last)
               + lam[i] * horizon_hours)
        c_t = min(max(c_t, c_last[i]), n_target)
        m = 100.0 * c_t / n_target if n_target > 0 else last_rate[i]
        lg = logit(m)
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        # Enforce the [0, 100] range and non-decreasing order after rounding.
        qs = [min(100.0, max(0.0, q)) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
