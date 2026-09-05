"""Evaluate model and baselines against the closing line — the real benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from metrics import summarize, log_loss, calibration_table

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"

BASE_FEATURES = ["elo_diff", "sp_runs_diff", "team_net_diff",
                 "home_sp_rest_capped", "vis_sp_rest_capped"]
MARKET_FEATURES = BASE_FEATURES + ["market_logit"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk_forward(df, feature_cols, label):
    seasons = sorted(df["season"].unique())
    preds = []
    for season in seasons[2:]:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        pipe.fit(train[feature_cols], train["home_win"])
        preds.append(pd.DataFrame({
            "game_id": test["game_id"].values,
            "season": season,
            label: pipe.predict_proba(test[feature_cols])[:, 1],
        }))
    return pd.concat(preds, ignore_index=True)


def main():
    df = pd.read_parquet(MODELING)
    df = df[df["p_market_home"].notna()].sort_values(["date", "game_id"]).reset_index(drop=True)
    df["market_logit"] = logit(df["p_market_home"])
    print(f"games with closing line: {len(df)}\n", flush=True)

    own = walk_forward(df, BASE_FEATURES, "p_own")
    plus = walk_forward(df, MARKET_FEATURES, "p_plus")

    ev = (df.merge(own, on=["game_id", "season"])
            .merge(plus, on=["game_id", "season"]))
    print(f"evaluated on {len(ev)} games ({ev['season'].min()}-{ev['season'].max()})\n", flush=True)

    rows = [
        summarize(ev["home_win"], [ev["home_win"].mean()] * len(ev), "base rate"),
        summarize(ev["home_win"], ev["p_elo"], "elo"),
        summarize(ev["home_win"], ev["p_own"], "own features"),
        summarize(ev["home_win"], ev["p_market_home"], "market (closing)"),
        summarize(ev["home_win"], ev["p_plus"], "own + market"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("\nmarket calibration:", flush=True)
    print(calibration_table(ev["home_win"], ev["p_market_home"]).to_string(), flush=True)

    print("\nlog loss by season:", flush=True)
    by = ev.groupby("season").apply(lambda g: pd.Series({
        "market": round(log_loss(g["home_win"], g["p_market_home"]), 5),
        "own": round(log_loss(g["home_win"], g["p_own"]), 5),
        "own+market": round(log_loss(g["home_win"], g["p_plus"]), 5),
    }), include_groups=False)
    print(by.to_string(), flush=True)

    beat = log_loss(ev["home_win"], ev["p_market_home"]) - log_loss(ev["home_win"], ev["p_plus"])
    print(f"\nown+market vs market alone (log loss gain): {beat:+.5f}", flush=True)


if __name__ == "__main__":
    main()