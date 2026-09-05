"""Bullpen workload features, computed point-in-time by calendar day.

MECHANISM: a bullpen that threw heavily over the previous two or three days has
fewer rested high-leverage arms available. Books price starters explicitly and
publicly; relief availability is not posted anywhere and is harder to observe.
If it is underweighted, it is a candidate edge.

Workload is rolled by CALENDAR DAY, not by game. An off day restores a bullpen;
a doubleheader depletes it twice. Counting "previous N games" would treat those
identically.

Relief pitches are total team pitches in a game minus that team's starter's
pitches, using the same first-pitch-of-the-half-inning starter identification
as build_pitcher_starts.py.

Every feature for a game is computed from strictly earlier games.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "processed" / "bullpen_features.parquet"

LOOKBACK_DAYS = [1, 2, 3]

TEAM_MAP = {
    "ARI": "ARI", "AZ": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CWS": "CHA", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCA",
    "LAA": "ANA", "LAD": "LAN", "MIA": "MIA", "FLA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYN", "NYY": "NYA",
    "OAK": "OAK", "ATH": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SDN", "SEA": "SEA",
    "SF": "SFN", "STL": "SLN", "TB": "TBA", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WAS",
}


def load_pitches():
    files = sorted(CACHE.glob("statcast_*.parquet"))
    if not files:
        raise SystemExit(f"No Statcast files in {CACHE}")
    cols = ["game_pk", "game_date", "pitcher", "inning_topbot",
            "home_team", "away_team", "at_bat_number", "pitch_number"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["game_date"])
    return df


def per_game_workload(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, pitching team): starter pitches, relief pitches."""
    # Starter = whoever threw the first pitch for that side.
    order = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    starters = (order.groupby(["game_pk", "inning_topbot"])
                     .first().reset_index()[["game_pk", "inning_topbot", "pitcher"]]
                     .rename(columns={"pitcher": "starter_id"}))
    df = df.merge(starters, on=["game_pk", "inning_topbot"], how="left")

    # 'Top' means the home team is pitching.
    is_home_pitching = df["inning_topbot"].str.lower().str.startswith("t")
    df["team_src"] = np.where(is_home_pitching, df["home_team"], df["away_team"])
    df["is_home"] = is_home_pitching
    df["is_relief"] = df["pitcher"] != df["starter_id"]

    g = df.groupby(["game_pk", "date", "team_src", "is_home"])
    out = g.agg(
        total_pitches=("pitch_number", "size"),
        relief_pitches=("is_relief", "sum"),
        n_pitchers=("pitcher", "nunique"),
    ).reset_index()
    out["n_relievers"] = out["n_pitchers"] - 1
    out["team"] = out["team_src"].map(TEAM_MAP)

    unmapped = sorted(set(out["team_src"]) - set(TEAM_MAP))
    if unmapped:
        print(f"UNMAPPED TEAM CODES: {unmapped}", flush=True)
    return out


def rolling_by_day(work: pd.DataFrame) -> pd.DataFrame:
    """For each game, relief workload over the prior N calendar days."""
    work = work.sort_values(["date", "game_pk"]).reset_index(drop=True)

    # history[team] -> list of (date, relief_pitches, n_relievers), appended
    # only after a row is emitted, so a game never sees itself.
    history = defaultdict(list)
    rows = []

    for r in work.itertuples(index=False):
        past = history[r.team]
        rec = {"game_pk": r.game_pk, "team": r.team, "is_home": r.is_home}

        for n in LOOKBACK_DAYS:
            cutoff = r.date - pd.Timedelta(days=n)
            pitches = sum(p for (d, p, _) in past if cutoff <= d < r.date)
            arms = sum(a for (d, _, a) in past if cutoff <= d < r.date)
            rec[f"bp_pitches_{n}d"] = pitches
            rec[f"bp_relievers_{n}d"] = arms

        # Days since this team last played -- an off day restores a bullpen.
        # Cap at 7: beyond that is the offseason or All-Star break, not rest.
        raw_gap = (r.date - past[-1][0]).days if past else np.nan
        rec["days_since_played"] = min(raw_gap, 7) if past else np.nan
        rows.append(rec)

        history[r.team].append((r.date, r.relief_pitches, r.n_relievers))

    return pd.DataFrame(rows)


def main():
    df = load_pitches()
    print(f"pitches loaded: {len(df):,}", flush=True)

    work = per_game_workload(df)
    print(f"team-games: {len(work):,}", flush=True)
    print("\nper-game relief workload:", flush=True)
    print(work[["total_pitches", "relief_pitches", "n_relievers"]]
          .describe().round(2).to_string(), flush=True)

    feat = rolling_by_day(work)

    # Pivot to one row per game, home and visitor side by side.
    home = feat[feat["is_home"]].drop(columns=["is_home", "team"]).add_prefix("home_")
    away = feat[~feat["is_home"]].drop(columns=["is_home", "team"]).add_prefix("vis_")
    home = home.rename(columns={"home_game_pk": "game_pk"})
    away = away.rename(columns={"vis_game_pk": "game_pk"})
    game = home.merge(away, on="game_pk", how="outer")

    # Differences oriented so positive favors the home team:
    # a more depleted VISITING bullpen helps the home side.
    for n in LOOKBACK_DAYS:
        game[f"bp_pitches_{n}d_diff"] = (
            game[f"vis_bp_pitches_{n}d"] - game[f"home_bp_pitches_{n}d"]
        )
    game["bp_rest_diff"] = game["home_days_since_played"] - game["vis_days_since_played"]

    print(f"\ngames: {len(game):,}", flush=True)
    cols = [f"bp_pitches_{n}d_diff" for n in LOOKBACK_DAYS] + [
        "home_bp_pitches_3d", "home_days_since_played", "bp_rest_diff"]
    print("\nfeature summary:", flush=True)
    print(game[cols].describe().round(3).to_string(), flush=True)
    print("\nmissing rate:", flush=True)
    print(game[cols].isna().mean().round(4).to_string(), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    game.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()