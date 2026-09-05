"""Is bullpen depletion a persistent state, or just noise?

The stabilization test used for pitcher metrics does not apply here: bullpen
workload is not a pitcher-season trait to be split in half. The equivalent
question for this feature family is whether depletion PERSISTS.

Three checks, all on team-games:

1. AUTOCORRELATION. Does relief workload over the prior 3 days predict relief
   workload in the next game? If a depleted bullpen simply gets used again at
   the same rate, "depletion" is not a state the model can exploit -- managers
   use whoever is available and the condition does not carry forward.

2. DOES IT PREDICT ANYTHING PHYSICAL? Workload should show up in observable
   consequences: fewer relievers used, or the starter being left in longer.
   If prior workload predicts neither, the feature is not measuring what the
   mechanism claims.

3. RAW ASSOCIATION WITH WINNING, before any modeling.

A feature that fails all three should not enter the model regardless of how
plausible its mechanism sounds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BULLPEN = ROOT / "data" / "processed" / "bullpen_features.parquet"
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
STARTS = ROOT / "data" / "processed" / "pitcher_starts.parquet"

LOCKBOX = [2024]


def team_game_frame():
    """Long format: one row per team-game, with prior workload and outcome."""
    bp = pd.read_parquet(BULLPEN)

    home = bp[[c for c in bp.columns if c.startswith("home_") or c == "game_pk"]].copy()
    home.columns = [c.replace("home_", "") for c in home.columns]
    home["is_home"] = True

    vis = bp[[c for c in bp.columns if c.startswith("vis_") or c == "game_pk"]].copy()
    vis.columns = [c.replace("vis_", "") for c in vis.columns]
    vis["is_home"] = False

    return pd.concat([home, vis], ignore_index=True)


def main():
    tg = team_game_frame()

    # Actual workload in the game itself, from the starts table.
    starts = pd.read_parquet(STARTS)[["game_pk", "is_home", "date", "team",
                                      "batters_faced", "pitches", "last_inning"]]
    starts["date"] = pd.to_datetime(starts["date"])
    starts = starts.rename(columns={"pitches": "sp_pitches"})

    df = tg.merge(starts, on=["game_pk", "is_home"], how="inner")
    df["season"] = df["date"].dt.year
    df = df[~df["season"].isin(LOCKBOX)]
    print(f"team-games (lockbox {LOCKBOX} withheld): {len(df):,}\n", flush=True)

    # --- 1. Autocorrelation: does prior workload predict next-game workload? ---
    df = df.sort_values(["team", "date", "game_pk"]).reset_index(drop=True)
    df["next_relief"] = df.groupby("team")["bp_pitches_1d"].shift(-1)

    print("1. PERSISTENCE", flush=True)
    print("   Does prior-N-day relief workload predict the NEXT game's", flush=True)
    print("   relief workload for the same team?\n", flush=True)
    for n in (1, 2, 3):
        col = f"bp_pitches_{n}d"
        sub = df[[col, "next_relief"]].dropna()
        r = sub[col].corr(sub["next_relief"])
        print(f"   prior {n}d workload -> next game workload: r = {r:+.4f} "
              f"(n={len(sub):,})", flush=True)

    # --- 2. Physical consequences ---
    print("\n2. OBSERVABLE CONSEQUENCES", flush=True)
    print("   Does prior workload predict how this game is managed?\n", flush=True)
    for col in ("bp_pitches_2d", "bp_pitches_3d", "days_since_played"):
        sub = df[[col, "sp_pitches", "last_inning"]].dropna()
        print(f"   {col}:", flush=True)
        print(f"      -> starter pitches:     r = "
              f"{sub[col].corr(sub['sp_pitches']):+.4f}", flush=True)
        print(f"      -> starter last inning: r = "
              f"{sub[col].corr(sub['last_inning']):+.4f}", flush=True)

    # --- 3. Raw association with winning ---
    mod = pd.read_parquet(MODELING)[["game_id", "home_win", "p_market_home", "season"]]
    bp = pd.read_parquet(BULLPEN)

    from velocity_edge import bridge_game_ids
    bp = bp.merge(bridge_game_ids(), on="game_pk", how="inner")
    m = mod.merge(bp, on="game_id", how="inner")
    m = m[~m["season"].isin(LOCKBOX)]

    print(f"\n3. RAW ASSOCIATION WITH WINNING  (n={len(m):,})", flush=True)
    print("   Positive means a more depleted VISITING bullpen helps the", flush=True)
    print("   home team, which is the direction the mechanism predicts.\n", flush=True)
    for n in (1, 2, 3):
        col = f"bp_pitches_{n}d_diff"
        sub = m[[col, "home_win"]].dropna()
        print(f"   {col}: r = {sub[col].corr(sub['home_win']):+.4f}", flush=True)
    sub = m[["bp_rest_diff", "home_win"]].dropna()
    print(f"   bp_rest_diff:       r = "
          f"{sub['bp_rest_diff'].corr(sub['home_win']):+.4f}", flush=True)

    print("\n   For comparison, features already in the model:", flush=True)
    ref = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    ref = ref[["game_id", "elo_diff", "team_net_diff", "home_win"]].dropna()
    print(f"   elo_diff:      r = {ref['elo_diff'].corr(ref['home_win']):+.4f}", flush=True)
    print(f"   team_net_diff: r = "
          f"{ref['team_net_diff'].corr(ref['home_win']):+.4f}", flush=True)


if __name__ == "__main__":
    main()