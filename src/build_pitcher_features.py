"""Point-in-time pitcher features, weighted by measured reliability.

Metric selection follows src/stabilization.py, which measured split-half
correlation at the 8-start window actually used here:

    velo_mean       0.978   physical measurement, near-perfect reliability
    k_rate          0.645   stabilized
    bb_rate         0.427   nearly stabilized
    xwoba_mean      0.392   marginal
    run_exp_per_bf  0.230   mostly noise at this window
    hr_rate         0.160   noise

run_exp_per_bf and hr_rate are deliberately excluded. Both describe past
performance well but are dominated by batted-ball luck at this sample size.

ERA BOUNDARY: xwOBA requires Statcast tracking and does not exist before 2015.
Release velocity, strikeouts, walks and run expectancy are available from 2010.
Every rolling metric therefore tracks its numerator and denominator as a
MATCHED PAIR -- a start missing a metric contributes to neither. Counting a
missing xwOBA as zero while still counting its batters would record every
pre-2015 start as a perfect performance.

Rate metrics are accumulated as summed numerators over summed denominators,
not as averages of per-start averages, so a 30-batter start weighs more than
a 12-batter one.

Velocity is carried as a deviation from the pitcher's own long-run baseline
rather than as a level: the level is reliable but says little about winning,
while a drop against a pitcher's own norm is a low-noise fatigue or injury
signal.
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STARTS = ROOT / "data" / "processed" / "pitcher_starts.parquet"
OUT = ROOT / "data" / "processed" / "pitcher_features.parquet"

WINDOW = 8          # recent starts, matching the stabilization analysis
BASELINE_MIN = 5    # starts required before a velocity baseline is trusted


def safe_div(num, den):
    return num / den if den else np.nan


def main():
    df = pd.read_parquet(STARTS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    print(f"starts: {len(df):,}", flush=True)

    # Every metric tracks numerator and denominator as a matched pair, so a
    # start with a missing metric contributes to neither side of the ratio.
    w = defaultdict(lambda: {
        "bf": deque(maxlen=WINDOW),
        "k": deque(maxlen=WINDOW),
        "bb": deque(maxlen=WINDOW),
        "xw": deque(maxlen=WINDOW), "xw_bf": deque(maxlen=WINDOW),
        "velo": deque(maxlen=WINDOW), "velo_pit": deque(maxlen=WINDOW),
    })
    base = defaultdict(lambda: {"velo_sum": 0.0, "pit_sum": 0.0, "n": 0})

    rows = []
    for r in df.itertuples(index=False):
        s = w[r.starter_id]
        b = base[r.starter_id]

        bf_sum = sum(s["bf"])
        n_prior = len(s["bf"])

        recent_velo = safe_div(sum(s["velo"]), sum(s["velo_pit"]))
        baseline_velo = (safe_div(b["velo_sum"], b["pit_sum"])
                         if b["n"] >= BASELINE_MIN else np.nan)

        rows.append({
            "game_pk": r.game_pk,
            "starter_id": r.starter_id,
            "is_home": r.is_home,
            "sp_prior_starts": n_prior,
            "sp_k_rate": safe_div(sum(s["k"]), bf_sum),
            "sp_bb_rate": safe_div(sum(s["bb"]), bf_sum),
            "sp_xwoba": safe_div(sum(s["xw"]), sum(s["xw_bf"])),
            "sp_bf_per_start": safe_div(bf_sum, n_prior),
            "sp_velo_recent": recent_velo,
            "sp_velo_delta": (recent_velo - baseline_velo)
                             if (pd.notna(recent_velo) and pd.notna(baseline_velo))
                             else np.nan,
        })

        # --- state updates happen only after the row is emitted ---
        s["bf"].append(r.batters_faced)
        s["k"].append(r.strikeouts)
        s["bb"].append(r.walks)

        if pd.notna(r.xwoba_mean):
            s["xw"].append(r.xwoba_mean * r.batters_faced)
            s["xw_bf"].append(r.batters_faced)

        if pd.notna(r.velo_mean):
            s["velo"].append(r.velo_mean * r.pitches)
            s["velo_pit"].append(r.pitches)
            b["velo_sum"] += r.velo_mean * r.pitches
            b["pit_sum"] += r.pitches
            b["n"] += 1

    feat = pd.DataFrame(rows)

    # Pivot to one row per game with home and visitor columns side by side.
    home = feat[feat["is_home"]].drop(columns=["is_home"]).add_prefix("home_")
    away = feat[~feat["is_home"]].drop(columns=["is_home"]).add_prefix("vis_")
    home = home.rename(columns={"home_game_pk": "game_pk"})
    away = away.rename(columns={"vis_game_pk": "game_pk"})

    game = home.merge(away, on="game_pk", how="outer")

    # Differences oriented so that positive favors the home team.
    game["sp_k_rate_diff"] = game["home_sp_k_rate"] - game["vis_sp_k_rate"]
    game["sp_bb_rate_diff"] = game["vis_sp_bb_rate"] - game["home_sp_bb_rate"]
    game["sp_xwoba_diff"] = game["vis_sp_xwoba"] - game["home_sp_xwoba"]
    game["sp_velo_delta_diff"] = game["home_sp_velo_delta"] - game["vis_sp_velo_delta"]

    print(f"games: {len(game):,}", flush=True)

    cols = ["sp_k_rate_diff", "sp_bb_rate_diff", "sp_xwoba_diff",
            "sp_velo_delta_diff", "home_sp_velo_delta", "home_sp_prior_starts"]
    print("\nsummary:", flush=True)
    print(game[cols].describe().round(4).to_string(), flush=True)

    print("\nmissing rate:", flush=True)
    print(game[cols].isna().mean().round(4).to_string(), flush=True)

    print("\nvelocity delta percentiles (should be tight around 0):", flush=True)
    print(game["home_sp_velo_delta"].quantile(
        [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]).round(3).to_string(), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    game.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()