"""Elo baseline, evaluated walk-forward: every game is predicted before it is used."""
from pathlib import Path

import pandas as pd

from metrics import summarize, calibration_table

ROOT = Path(__file__).resolve().parent.parent
SPINE = ROOT / "data" / "processed" / "spine.parquet"
OUT = ROOT / "data" / "processed" / "predictions_elo.parquet"

K = 4.0                # update size; baseball has low signal per game
HOME_EDGE = 24.0       # in Elo points
CARRYOVER = 0.75       # season-to-season regression toward 1500
START = 1500.0


def expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def run_elo(df: pd.DataFrame) -> pd.DataFrame:
    ratings = {}
    prev_season = None
    preds = []

    for row in df.itertuples(index=False):
        if prev_season is not None and row.season != prev_season:
            for t in ratings:
                ratings[t] = START + CARRYOVER * (ratings[t] - START)
        prev_season = row.season

        home = ratings.setdefault(row.home_team, START)
        vis = ratings.setdefault(row.vis_team, START)

        # Prediction uses only ratings built from strictly earlier games.
        p_home = expected(home + HOME_EDGE, vis)
        preds.append(p_home)

        result = float(row.home_win)
        delta = K * (result - p_home)
        ratings[row.home_team] = home + delta
        ratings[row.vis_team] = vis - delta

    out = df.copy()
    out["p_elo"] = preds
    return out


def main():
    df = pd.read_parquet(SPINE).sort_values(["date", "game_id"]).reset_index(drop=True)
    df = run_elo(df)

    # Burn in the first season: ratings start at a uniform 1500 and mean nothing yet.
    first_season = df["season"].min()
    ev = df[df["season"] > first_season]
    print(f"evaluating {len(ev)} games ({first_season + 1}-{df['season'].max()})\n", flush=True)

    rows = [
        summarize(ev["home_win"], [0.5] * len(ev), "coin flip"),
        summarize(ev["home_win"], [ev["home_win"].mean()] * len(ev), "base rate (home)"),
        summarize(ev["home_win"], ev["p_elo"], "elo"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("\nelo calibration:", flush=True)
    print(calibration_table(ev["home_win"], ev["p_elo"]).to_string(), flush=True)

    print("\nelo log loss by season:", flush=True)
    from metrics import log_loss
    print(ev.groupby("season").apply(
        lambda g: round(log_loss(g["home_win"], g["p_elo"]), 5), include_groups=False
    ).to_string(), flush=True)

    df.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()