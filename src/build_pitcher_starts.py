"""Aggregate Statcast pitches into one row per starting pitcher per game."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "processed" / "pitcher_starts.parquet"

# Statcast abbreviation -> Retrosheet code.
TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CWS": "CHA", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCA",
    "LAA": "ANA", "LAD": "LAN", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYN", "NYY": "NYA", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SDN", "SEA": "SEA",
    "SF": "SFN", "STL": "SLN", "TB": "TBA", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WAS", "ARI": "ARI", "AZ": "ARI",
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "PHI": "PHI", 
    "PIT": "PIT", "SD": "SDN", "SEA": "SEA", "ATH": "OAK",
}

STRIKEOUTS = {"strikeout", "strikeout_double_play"}
WALKS = {"walk", "hit_by_pitch"}
HITS = {"single", "double", "triple", "home_run"}


def identify_starters(df: pd.DataFrame) -> pd.DataFrame:
    """The starter is whoever threw the first pitch for their side of the game."""
    order = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first = order.groupby(["game_pk", "inning_topbot"]).first().reset_index()
    return first[["game_pk", "inning_topbot", "pitcher"]].rename(
        columns={"pitcher": "starter_id"}
    )


def main():
    files = sorted(CACHE.glob("statcast_*.parquet"))
    if not files:
        raise SystemExit(f"No Statcast files in {CACHE}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"pitches loaded: {len(df):,}", flush=True)

    starters = identify_starters(df)
    df = df.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    sp = df[df["pitcher"] == df["starter_id"]].copy()
    print(f"pitches by starters: {len(sp):,}  ({len(sp) / len(df):.1%})", flush=True)

    # inning_topbot marks which half the pitcher is throwing in:
    # 'Top' means the home team is pitching, 'Bot' means the away team is.
    sp["pitching_team_src"] = np.where(
        sp["inning_topbot"].str.lower().str.startswith("t"),
        sp["home_team"], sp["away_team"]
    )
    sp["is_home"] = sp["inning_topbot"].str.lower().str.startswith("t")

    ev = sp["events"]
    sp["is_k"] = ev.isin(STRIKEOUTS)
    sp["is_bb"] = ev.isin(WALKS)
    sp["is_hit"] = ev.isin(HITS)
    sp["is_hr"] = ev == "home_run"
    sp["is_pa"] = ev.notna()

    g = sp.groupby(["game_pk", "game_date", "pitching_team_src", "is_home", "starter_id"])
    starts = g.agg(
        pitcher_name=("player_name", "first"),
        pitches=("pitch_number", "size"),
        batters_faced=("is_pa", "sum"),
        outs_recorded=("is_pa", "size"),          # placeholder, replaced below
        strikeouts=("is_k", "sum"),
        walks=("is_bb", "sum"),
        hits=("is_hit", "sum"),
        home_runs=("is_hr", "sum"),
        xwoba_mean=("estimated_woba_using_speedangle", "mean"),
        run_exp_total=("delta_run_exp", "sum"),
        last_inning=("inning", "max"),
        velo_mean=("release_speed", "mean"),
    ).reset_index()

    # Outs are not directly available per pitch; approximate innings from the
    # last inning the starter appeared in. This is a known approximation.
    starts = starts.drop(columns=["outs_recorded"])

    starts["k_rate"] = starts["strikeouts"] / starts["batters_faced"]
    starts["bb_rate"] = starts["walks"] / starts["batters_faced"]
    starts["run_exp_per_bf"] = starts["run_exp_total"] / starts["batters_faced"]
    starts["team"] = starts["pitching_team_src"].map(TEAM_MAP)
    starts["date"] = pd.to_datetime(starts["game_date"])

    unmapped = sorted(set(starts["pitching_team_src"]) - set(TEAM_MAP))
    if unmapped:
        print(f"\nUNMAPPED TEAM CODES: {unmapped}", flush=True)

    print(f"\nstarts: {len(starts):,}", flush=True)
    print(f"games covered: {starts['game_pk'].nunique():,} "
          f"(expect 2 starts per game)", flush=True)

    per_game = starts.groupby("game_pk").size().value_counts()
    print(f"\nstarters per game:\n{per_game.to_string()}", flush=True)

    cols = ["pitches", "batters_faced", "k_rate", "bb_rate",
            "xwoba_mean", "run_exp_per_bf", "last_inning", "velo_mean"]
    print("\nsummary:", flush=True)
    print(starts[cols].describe().round(3).to_string(), flush=True)

    print("\nbest 10 starts by run expectancy prevented:", flush=True)
    show = ["date", "pitcher_name", "team", "batters_faced", "strikeouts",
            "run_exp_total", "xwoba_mean"]
    print(starts.nsmallest(10, "run_exp_total")[show].to_string(index=False), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    starts.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()