"""Walk-forward logistic regression, evaluated season by season against Elo."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from metrics import summarize, calibration_table

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "data" / "processed" / "predictions_model.parquet"

FEATURE_COLS = [
    "elo_diff",
    "sp_runs_diff",
    "team_net_diff",
    "home_sp_rest_capped",
    "vis_sp_rest_capped",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Long gaps are offseason or injury, not rest. Cap so they can't dominate.
    for side in ("home", "vis"):
        df[f"{side}_sp_rest_capped"] = df[f"{side}_sp_rest"].clip(upper=10)
    return df


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    preds = []

    # Need at least two prior seasons of history before the first prediction.
    for season in seasons[2:]:
        train = df[df["season"] < season]
        test = df[df["season"] == season]

        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        pipe.fit(train[FEATURE_COLS], train["home_win"])
        p = pipe.predict_proba(test[FEATURE_COLS])[:, 1]

        preds.append(pd.DataFrame({
            "game_id": test["game_id"].values,
            "season": season,
            "home_win": test["home_win"].values,
            "p_elo": test["p_elo"].values,
            "p_model": p,
        }))

        coefs = dict(zip(FEATURE_COLS, pipe.named_steps["clf"].coef_[0].round(4)))
        print(f"  {season}: trained on {len(train):5d} games  {coefs}", flush=True)

    return pd.concat(preds, ignore_index=True)


def main():
    df = prepare(pd.read_parquet(FEATURES)).sort_values(["date", "game_id"])
    print("fitting walk-forward by season:", flush=True)
    ev = walk_forward(df)

    print(f"\nevaluated on {len(ev)} games ({ev['season'].min()}-{ev['season'].max()})\n", flush=True)
    rows = [
        summarize(ev["home_win"], [ev["home_win"].mean()] * len(ev), "base rate"),
        summarize(ev["home_win"], ev["p_elo"], "elo"),
        summarize(ev["home_win"], ev["p_model"], "logistic"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("\nlog loss by season (elo vs model):", flush=True)
    from metrics import log_loss
    by_season = ev.groupby("season").apply(
        lambda g: pd.Series({
            "elo": round(log_loss(g["home_win"], g["p_elo"]), 5),
            "model": round(log_loss(g["home_win"], g["p_model"]), 5),
        }), include_groups=False
    )
    by_season["improvement"] = (by_season["elo"] - by_season["model"]).round(5)
    print(by_season.to_string(), flush=True)

    print("\nmodel calibration:", flush=True)
    print(calibration_table(ev["home_win"], ev["p_model"]).to_string(), flush=True)

    ev.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()