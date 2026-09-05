"""Point-in-time park run environment.

A park factor is computed for every game from games played at that park
STRICTLY EARLIER, expanding across all prior history. Park factors change --
fences move, humidors get installed, the ball changes -- so a fixed lookup
table computed over the full dataset would leak the future into the past.

SHRINKAGE: a park with few prior games gets pulled toward league average. The
raw ratio of runs at a park to league average is extremely noisy early in a
park's history, and an unshrunk factor would hand the model a number that is
mostly sampling error. The shrinkage weight is n / (n + K) with K = 200 games,
roughly 1.3 seasons of home games.

MECHANISM for using this at all (stated before testing): park does not favor
the home team -- both sides bat in it. What a hitter's park changes is
VARIANCE. Higher run environments widen the distribution of run differential,
which compresses win probability toward .500. If the market does not fully
adjust, favorites should win LESS often than the closing line implies in
high-run parks. This is a second-order effect on the shape of the outcome
distribution rather than a first-order effect on team strength.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPINE = ROOT / "data" / "processed" / "spine.parquet"
OUT = ROOT / "data" / "processed" / "park_features.parquet"

SHRINK_K = 200      # prior games before a park factor is trusted at full weight
MIN_GAMES = 30      # below this, emit NaN rather than a heavily shrunk guess


def main():
    df = pd.read_parquet(SPINE).sort_values(["date", "game_id"]).reset_index(drop=True)
    df["total_runs"] = df["home_score"] + df["vis_score"]
    print(f"games: {len(df):,}", flush=True)
    print(f"parks: {df['park_id'].nunique()}", flush=True)
    print(f"league runs per game: {df['total_runs'].mean():.3f}\n", flush=True)

    park_runs = defaultdict(float)
    park_games = defaultdict(int)
    league_runs = 0.0
    league_games = 0

    rows = []
    for r in df.itertuples(index=False):
        n = park_games[r.park_id]
        league_mean = league_runs / league_games if league_games else np.nan

        if n >= MIN_GAMES and league_games > 0 and league_mean > 0:
            raw = (park_runs[r.park_id] / n) / league_mean
            w = n / (n + SHRINK_K)
            factor = 1.0 + w * (raw - 1.0)
        else:
            raw = np.nan
            factor = np.nan

        rows.append({
            "game_id": r.game_id,
            "park_id": r.park_id,
            "park_prior_games": n,
            "park_factor_raw": raw,
            "park_factor": factor,
            "league_rpg_to_date": league_mean,
        })

        # --- state updates only after the row is emitted ---
        park_runs[r.park_id] += r.total_runs
        park_games[r.park_id] += 1
        league_runs += r.total_runs
        league_games += 1

    out = pd.DataFrame(rows)
    out = out.merge(df[["game_id", "season", "total_runs"]], on="game_id", how="left")

    print("park factor summary:", flush=True)
    print(out[["park_factor_raw", "park_factor", "park_prior_games"]]
          .describe().round(4).to_string(), flush=True)
    print(f"\nmissing park_factor: {out['park_factor'].isna().mean():.2%} "
          f"(parks with <{MIN_GAMES} prior games)", flush=True)

    # Does the factor actually predict runs in the game it describes?
    sub = out[["park_factor", "total_runs"]].dropna()
    print(f"\npark_factor vs actual total runs in that game: "
          f"r = {sub['park_factor'].corr(sub['total_runs']):+.4f}  "
          f"(n={len(sub):,})", flush=True)

    print("\nmost extreme parks by final factor "
          "(parks with 500+ games):", flush=True)
    last = (out[out["park_prior_games"] >= 500]
            .sort_values("park_prior_games")
            .groupby("park_id").last()
            .sort_values("park_factor"))
    show = last[["park_prior_games", "park_factor_raw", "park_factor"]].round(4)
    print(pd.concat([show.head(5), show.tail(5)]).to_string(), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()