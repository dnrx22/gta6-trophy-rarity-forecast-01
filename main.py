#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads every <id>.in case in MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out forecasts to MAKEFUN_OUTPUT_DIR (default /output).

Model (stdlib only, no network, deterministic):
  * Work on the logit scale, logit(p) = ln(p / (100 - p)) with rates in percent.
  * The Exophase page is a stale snapshot: it sits unchanged for hours and
    then jumps at a refresh, when newly tracked players dilute most rates.
    So history reads are first collapsed into plateaus (consecutive identical
    reads carry no new information).
  * When the plateaus show real movement, fit a per-trophy weighted
    least-squares line of logit(rate) against ln(1 + hours since release),
    shrink each slope toward its tier mean, anchor the forecast at the last
    observed rate, and clamp it between a counts floor (earners so far divided
    by the forecast players-tracked) and a launch-week ceiling that depends on
    the trophy's kind (story chapter, grind, completion, ...).
  * When the history is perfectly flat and the snapshot predates launch, the
    next refresh is still ahead of the target: forecast a two-component
    mixture of "page stays frozen" and "one refresh dilutes the rates",
    with the refresh probability from a hazard model on how stale the page
    already is.  This is the launch-night regime of the practice cases.
  * Spreads widen honestly with horizon and with how noisy the fit is.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from statistics import NormalDist

NORM = NormalDist()
Z = tuple(NORM.inv_cdf(level) for level in (0.05, 0.25, 0.5, 0.75, 0.95))

RELEASE_HOUR = None  # set per case from game.release_date (midnight UTC)

RATE_LO = 0.01
RATE_HI = 99.99


def clamp_rate(p):
    """Clamp a percent rate into the range the logit is defined on."""
    return min(max(float(p), RATE_LO), RATE_HI)


def logit(p):
    p = clamp_rate(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(L):
    if L > 30.0:
        return 100.0
    if L < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-L))


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hours_since_release(moment, release):
    seconds = (moment - release).total_seconds()
    return max(seconds, 0.0) / 3600.0


# --------------------------------------------------------------------------
# Trophy classification: launch-week ceiling and first-refresh earn factor.
# A trophy's week-one ceiling is the share of tracked players who can
# plausibly hold it seven days after launch; the refresh factor is how much
# its earner count grows when the page first refreshes after launch (the
# opening trophy is earned within minutes, everything else barely moves).
# --------------------------------------------------------------------------

def classify(trophy):
    """Return (ceiling_pct, refresh_earn_factor) for one trophy."""
    tier = trophy.get("tier", "bronze")
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc

    # The opening trophy: every newly tracked player holds it already.
    if "gone fishing" in text:
        return 99.5, 6.06

    if tier == "platinum":
        return 2.0, 1.05
    if "100% completion" in text:
        return 1.5, 1.05
    if tier == "gold":  # the final mission
        return 12.0, 1.30

    chapter = re.search(r"complete chapter (\d+)", desc)
    if chapter:
        ceilings = {1: 55.0, 2: 45.0, 3: 38.0, 4: 32.0}
        return ceilings.get(int(chapter.group(1)), 30.0), 1.30
    if "complete the final mission for" in text:
        return 30.0, 1.25
    if "complete" in desc and trophy.get("secret"):
        return 35.0, 1.30

    if tier == "silver":
        if "250 stunts" in text:
            return 8.0, 1.10
        if "gold medal" in text:
            return 6.0, 1.10
        if "mission strands for strangers" in text:
            return 20.0, 1.15
        if "collectible card sets" in text:
            return 5.0, 1.10
        if "100 locations" in text:
            return 10.0, 1.10
        if "bank vaults" in text:
            return 12.0, 1.15
        return 12.0, 1.15

    # Bronze open-world / misc trophies, roughly by how early they come.
    easy = [
        ("try each type of activity", 35.0),
        ("gone fishing", 99.5),
        ("complete an arcade game", 30.0),
        ("beat cal in a fishing", 30.0),
        ("explore all of leonida", 30.0),
        ("discover 100 vehicles", 25.0),
        ("rob 3 shops", 25.0),
        ("take down 5 enemies", 25.0),
        ("enjoy a cocktail", 25.0),
        ("rob an armored truck", 25.0),
        ("spend $100,000", 20.0),
        ("lose a 4 star wanted level", 20.0),
        ("rob a jewelry store", 20.0),
        ("bowling alley", 20.0),
        ("register 10 vehicles", 20.0),
        ("second time's the charm", 20.0),
        ("choices made", 25.0),
        ("make an unscheduled departure", 20.0),
        ("smash your way out", 20.0),
        ("hunt a unique animal", 15.0),
        ("kinda hot in these rhinos", 15.0),
        ("partnership with mass appeal", 15.0),
        ("what the hell just happened", 15.0),
        ("bowl 'em over", 15.0),
        ("rob an illegal operation", 15.0),
        ("win 20 different races", 12.0),
        ("complete 50 challenges", 10.0),
        ("purchase all garage properties", 10.0),
        ("win a top fun drone race", 10.0),
        ("meditate at every spot", 8.0),
        ("laundering marked bills", 8.0),
    ]
    for needle, ceiling in easy:
        if needle in text:
            return ceiling, 1.20
    return 20.0, 1.20


# --------------------------------------------------------------------------
# History preparation: collapse consecutive identical reads into plateaus.
# --------------------------------------------------------------------------

def dedupe_plateaus(history):
    """Keep the first read of each run of identical (tracked, rates) reads."""
    plateaus = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            plateaus.append(read)
            prev_key = key
    return plateaus


def trophy_series(plateaus, index, release):
    """Return (times_hours, x, y, weights) for one trophy's logit series."""
    if not plateaus:
        return [], [], [], []
    last_slot = parse_time(plateaus[-1]["slot"])
    xs, ys, ws = [], [], []
    for read in plateaus:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        slot = parse_time(read["slot"])
        age_hours = max((last_slot - slot).total_seconds(), 0.0) / 3600.0
        xs.append(math.log1p(hours_since_release(slot, release)))
        ys.append(logit(rates[index]))
        ws.append(math.exp(-age_hours / 24.0))
    return xs, ys, ws


def weighted_line(xs, ys, ws):
    """Weighted least squares line; returns (intercept, slope, resid_std)."""
    total_w = sum(ws)
    if total_w <= 0.0 or len(xs) < 2:
        return None
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        / total_w
    )
    return intercept, slope, resid


# --------------------------------------------------------------------------
# Flat-history branch: mixture of "frozen" and "one refresh dilutes".
# --------------------------------------------------------------------------

def mixture_quantiles(mu_a, sd_a, mu_b, sd_b, weight_b, levels):
    """Quantiles of a two-normal mixture on the logit scale, by grid search."""
    lo = min(mu_a - 6.0 * sd_a, mu_b - 6.0 * sd_b)
    hi = max(mu_a + 6.0 * sd_a, mu_b + 6.0 * sd_b)
    dist_a = NormalDist(mu_a, sd_a)
    dist_b = NormalDist(mu_b, sd_b)
    step = 0.01
    grid = []
    value = lo
    while value <= hi:
        cdf = (1.0 - weight_b) * dist_a.cdf(value) + weight_b * dist_b.cdf(value)
        grid.append((value, cdf))
        value += step
    out = []
    for level in levels:
        found = grid[-1][0]
        for value, cdf in grid:
            if cdf >= level:
                found = value
                break
        out.append(found)
    return out


def flat_history_forecast(trophy, last_rate, ceiling, earn_factor,
                          stale_hours, horizon_hours, levels):
    """Forecast when the page has never moved: expect a refresh may land."""
    # Refresh hazard: the page's counts refresh every few hours, not
    # continuously (the house saw one gap of about four and a half hours on
    # launch night). Model the gap as roughly Normal(4.5 h, 0.8 h) and take
    # P(a refresh lands before the target | none has landed for stale_hours).
    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    gap_mean, gap_sd = 4.5, 0.8
    p_no_refresh_yet = 1.0 - Phi((stale_hours - gap_mean) / gap_sd)
    p_refresh = (Phi((stale_hours + horizon_hours - gap_mean) / gap_sd)
                 - Phi((stale_hours - gap_mean) / gap_sd))
    p_refresh = p_refresh / max(p_no_refresh_yet, 1e-6)
    p_refresh = min(max(p_refresh, 0.02), 0.95)

    mu_flat = logit(last_rate)
    if last_rate >= 99.5:
        # The opening trophy: every newly tracked player holds it already.
        diluted = last_rate
    else:
        diluted = clamp_rate(last_rate * earn_factor / 6.06)
    mu_diluted = logit(diluted)

    logits = mixture_quantiles(mu_flat, 0.10, mu_diluted, 0.45, p_refresh, levels)
    return [clamp_rate(inv_logit(L)) for L in logits]


# --------------------------------------------------------------------------
# Fallback: last rate with a random-walk spread (used only on bad input).
# --------------------------------------------------------------------------

def fallback_row(last_rate, horizon_days, levels):
    sigma = 0.5 * math.sqrt(max(horizon_days, 0.0))
    centre = logit(last_rate)
    return [clamp_rate(inv_logit(centre + z * sigma)) for z in Z]


# --------------------------------------------------------------------------
# Per-case forecast.
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]

    release_date = (case.get("game") or {}).get("release_date", "2026-11-19")
    release = datetime.fromisoformat(release_date).replace(tzinfo=timezone.utc)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max((target - at).total_seconds(), 0.0) / 86400.0
    horizon_hours = horizon_days * 24.0

    plateaus = dedupe_plateaus(history)
    last = plateaus[-1] if plateaus else None
    last_rates = (last or {}).get("rates") or []
    tracked_last = (last or {}).get("players_tracked") or 0

    # Does the history show any movement at all?
    moved = False
    if len(plateaus) >= 2:
        first_rates = plateaus[0].get("rates") or []
        for a, b in zip(first_rates, last_rates):
            if a is not None and b is not None and a != b:
                moved = True
                break
        if (plateaus[0].get("players_tracked") != tracked_last):
            moved = True

    # Forecast the players-tracked count at the target (log-time fit).
    tracked_target = float(tracked_last) if tracked_last else 1.0
    if len(plateaus) >= 3 and tracked_last:
        txs, tys, tws = [], [], []
        last_slot = parse_time(plateaus[-1]["slot"])
        for read in plateaus:
            tracked = read.get("players_tracked")
            if not tracked:
                continue
            slot = parse_time(read["slot"])
            age = max((last_slot - slot).total_seconds(), 0.0) / 3600.0
            txs.append(math.log1p(hours_since_release(slot, release)))
            tys.append(math.log(float(tracked)))
            tws.append(math.exp(-age / 24.0))
        fit = weighted_line(txs, tys, tws)
        if fit is not None:
            _, slope_t, _ = fit
            if slope_t > 0.0:
                x_target = math.log1p(hours_since_release(target, release))
                x_last = math.log1p(hours_since_release(last_slot, release))
                tracked_target = math.exp(
                    math.log(float(tracked_last)) + slope_t * (x_target - x_last)
                )
                tracked_target = min(max(tracked_target, float(tracked_last)),
                                     float(tracked_last) * 50.0)

    # Per-trophy trend fits, then shrink slopes toward their tier mean.
    raw_slopes = {}
    residuals = {}
    for idx, trophy in enumerate(trophies):
        xs, ys, ws = trophy_series(plateaus, idx, release)
        fit = weighted_line(xs, ys, ws) if len(xs) >= 3 else None
        if fit is not None:
            _, slope, resid = fit
            raw_slopes[idx] = max(min(slope, 3.0), -3.0)
            residuals[idx] = resid
        else:
            raw_slopes[idx] = 0.0
            residuals[idx] = 0.0

    tier_slopes = {}
    for idx, trophy in enumerate(trophies):
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(raw_slopes[idx])
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    global_mean = sum(raw_slopes.values()) / max(len(raw_slopes), 1)

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = None
        if idx < len(last_rates) and last_rates[idx] is not None:
            last_rate = float(last_rates[idx])
        if last_rate is None:
            # No read ever listed this trophy: fall back to a middling rate.
            last_rate = 1.0
        ceiling, earn_factor = classify(trophy)

        if not moved and plateaus:
            stale_hours = 0.0
            if len(history) >= 1:
                first_slot = parse_time(history[0]["slot"])
                stale_hours = max((at - first_slot).total_seconds(), 0.0) / 3600.0
            qs = flat_history_forecast(
                trophy, last_rate, ceiling, earn_factor,
                stale_hours, horizon_hours, levels,
            )
            rows.append({"id": trophy["id"], "q": [round(q, 4) for q in qs]})
            continue

        # Trend branch: shrunk slope anchored at the last observed rate.
        n_plat = len(plateaus)
        shrink = n_plat / (n_plat + 3.0)
        tier = trophy.get("tier", "bronze")
        prior_slope = tier_mean.get(tier, global_mean)
        slope = shrink * raw_slopes[idx] + (1.0 - shrink) * prior_slope

        last_slot = parse_time(plateaus[-1]["slot"]) if plateaus else at
        x_target = math.log1p(hours_since_release(target, release))
        x_last = math.log1p(hours_since_release(last_slot, release))
        dx = max(x_target - x_last, 0.0)

        # Counts floor: earners so far cannot un-earn; rates cannot fall
        # below earners-so-far divided by the forecast tracked count.
        floor_rate = RATE_LO
        if tracked_last and tracked_target > 0:
            earned_last = last_rate * tracked_last / 100.0
            floor_rate = 100.0 * earned_last / tracked_target
        # Launch-week ceiling: the rate cannot run past what a week of
        # play plausibly gives this kind of trophy. The ceiling never
        # forces a forecast below the rate already observed.
        ceiling = max(ceiling, last_rate)

        # Saturating trend: the fitted slope is the local velocity, but a
        # straight line in log-time would run past the ceiling (or the
        # floor) forever. Instead close the gap to the ceiling / floor
        # exponentially in log-time: the gap shrinks by the fitted rate,
        # so the forecast decelerates as it approaches its bound and
        # agrees with the linear fit over short horizons.
        y_last = logit(last_rate)
        if slope > 0.0:
            gap = logit(ceiling) - y_last
            if gap <= 0.05:
                centre = y_last
            else:
                rate_k = slope / gap
                centre = logit(ceiling) - gap * math.exp(-rate_k * dx)
        elif slope < 0.0:
            gap = y_last - logit(floor_rate)
            if gap <= 0.05:
                centre = y_last
            else:
                rate_k = -slope / gap
                centre = logit(floor_rate) + gap * math.exp(-rate_k * dx)
        else:
            centre = y_last

        resid = residuals.get(idx, 0.0)
        sigma = math.sqrt(
            0.13 ** 2
            + min(resid, 1.0) ** 2
            + 0.08 * horizon_days
            + (0.20 * abs(slope) * dx) ** 2
        )
        sigma = min(max(sigma, 0.22), 1.5)

        qs = [clamp_rate(inv_logit(centre + z * sigma)) for z in Z]
        qs = [round(q, 4) for q in qs]
        # Keep the quantiles non-decreasing after rounding.
        for k in range(1, len(qs)):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception:
            # Never leave a case without a valid forecast.
            horizon_days = 0.0
            try:
                at = parse_time(case["at"])
                target = parse_time(case["target"])
                horizon_days = max((target - at).total_seconds(), 0.0) / 86400.0
            except Exception:
                pass
            history = case.get("history") or []
            last_rates = (history[-1].get("rates") or []) if history else []
            levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
            rows = []
            for trophy in case["trophies"]:
                idx = case["trophies"].index(trophy)
                rate = 1.0
                if idx < len(last_rates) and last_rates[idx] is not None:
                    rate = float(last_rates[idx])
                rows.append({"id": trophy["id"],
                             "q": [round(q, 4) for q in
                                   fallback_row(rate, horizon_days, levels)]})
            rows.sort(key=lambda row: row["id"])
            result = {"trophies": rows}
        with open(os.path.join(output_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
