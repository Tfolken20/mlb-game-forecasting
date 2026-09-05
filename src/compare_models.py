"""Does model class matter, or is the feature set the binding constraint?

Compares three model families under identical walk-forward evaluation:

    logistic     regularized linear, the current baseline
    gbm          histogram gradient boosting, for nonlinearity and interactions
    stack        logistic meta-learner over both base models

STACKING UNDER WALK-FORWARD REQUIRES NESTED SPLITS.

The meta-learner must be trained on base-model predictions for games the base
models did not see. Fitting base and meta on the same rows teaches the meta
to trust an overfit signal -- excellent in sample, worse out of sample.

For each target season S:
    1. Inner walk-forward across the training seasons produces out-of-fold
       base predictions.
    2. The meta-learner is fit on those out-of-fold predictions only.
    3. Base models are refit on all seasons < S and predict S.
    4. The meta-learner combines those predictions.

Nothing from season S touches any fit.

EXPECTATION, stated before running: gradient boosting should gain little.
Seven features with mostly linear, monotone relationships to the outcome give
a tree ensemble little to exploit, and the earlier reliability analysis
suggests the constraint is signal in the features rather than model capacity.
Reported either way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import summarize, log_loss, calibration_table
from velocity_edge import bridge_game_ids

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
PITCHER = ROOT / "data" / "processed" / "pitcher_features.parquet"

LOCKBOX = [2024]

FEATURES = [
    "elo_diff", "team_net_diff",
    "home_sp_rest_capped", "vis_sp_rest_capped",
    "sp_k_rate_diff", "sp_bb_rate_diff", "sp_velo_delta_diff",
]

MIN_TRAIN_SEASONS = 2      # before the first prediction
MIN_META_SEASONS = 4       # before stacking has enough out-of-fold data


def make_logistic():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def make_gbm():
    # Conservative settings: this dataset is small and the signal is weak,
    # so depth and leaf count are held down to limit variance.
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=3,
        max_leaf_nodes=8,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=7,
    )


def fit_predict(model_fn, train, test):
    m = model_fn()
    m.fit(train[FEATURES], train["home_win"])
    return m.predict_proba(test[FEATURES])[:, 1]


def out_of_fold(df, seasons):
    """Inner walk-forward: base predictions for seasons the base never saw."""
    rows = []
    for i, s in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        inner_train = df[df["season"].isin(seasons[:i])]
        inner_test = df[df["season"] == s]
        if len(inner_test) == 0:
            continue
        rows.append(pd.DataFrame({
            "home_win": inner_test["home_win"].values,
            "p_log": fit_predict(make_logistic, inner_train, inner_test),
            "p_gbm": fit_predict(make_gbm, inner_train, inner_test),
        }))
    return pd.concat(rows, ignore_index=True) if rows else None


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    df = pd.read_parquet(MODELING)
    pf = pd.read_parquet(PITCHER).merge(bridge_game_ids(), on="game_pk", how="inner")
    df = df.merge(pf.drop(columns=["game_pk"]), on="game_id", how="left")
    df = df[df["p_market_home"].notna()]

    held = df[df["season"].isin(LOCKBOX)]
    df = df[~df["season"].isin(LOCKBOX)]
    print(f"LOCKBOX: {LOCKBOX} withheld ({len(held):,} games, not examined)", flush=True)

    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    print(f"seasons: {seasons}", flush=True)

    preds = []
    for i, s in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == s]

        p_log = fit_predict(make_logistic, train, test)
        p_gbm = fit_predict(make_gbm, train, test)

        p_stack = np.full(len(test), np.nan)
        if i >= MIN_META_SEASONS:
            oof = out_of_fold(df, seasons[:i])
            if oof is not None and len(oof) > 500:
                meta = LogisticRegression(C=1.0, max_iter=1000)
                meta.fit(
                    np.column_stack([logit(oof["p_log"]), logit(oof["p_gbm"])]),
                    oof["home_win"],
                )
                p_stack = meta.predict_proba(
                    np.column_stack([logit(p_log), logit(p_gbm)])
                )[:, 1]

        preds.append(pd.DataFrame({
            "game_id": test["game_id"].values,
            "season": s,
            "home_win": test["home_win"].values,
            "p_market_home": test["p_market_home"].values,
            "p_log": p_log,
            "p_gbm": p_gbm,
            "p_stack": p_stack,
        }))
        print(f"  {s}: trained on {len(train):,}, predicted {len(test):,}", flush=True)

    ev = pd.concat(preds, ignore_index=True)
    print(f"\nevaluated: {len(ev):,} games "
          f"({ev['season'].min()}-{ev['season'].max()})\n", flush=True)

    rows = [
        summarize(ev["home_win"], ev["p_log"], "logistic"),
        summarize(ev["home_win"], ev["p_gbm"], "gbm"),
        summarize(ev["home_win"], ev["p_market_home"], "market"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    st = ev[ev["p_stack"].notna()]
    if len(st):
        print(f"\nstacking evaluated on {len(st):,} games "
              f"({st['season'].min()}-{st['season'].max()}):", flush=True)
        srows = [
            summarize(st["home_win"], st["p_log"], "logistic"),
            summarize(st["home_win"], st["p_gbm"], "gbm"),
            summarize(st["home_win"], st["p_stack"], "stack"),
            summarize(st["home_win"], st["p_market_home"], "market"),
        ]
        print(pd.DataFrame(srows).to_string(index=False), flush=True)

    print("\nlog loss by season:", flush=True)
    by = ev.groupby("season").apply(lambda g: pd.Series({
        "logistic": round(log_loss(g["home_win"], g["p_log"]), 5),
        "gbm": round(log_loss(g["home_win"], g["p_gbm"]), 5),
        "stack": (round(log_loss(g["home_win"], g["p_stack"]), 5)
                  if g["p_stack"].notna().all() else np.nan),
        "market": round(log_loss(g["home_win"], g["p_market_home"]), 5),
    }), include_groups=False)
    print(by.to_string(), flush=True)

    gain = log_loss(ev["home_win"], ev["p_log"]) - log_loss(ev["home_win"], ev["p_gbm"])
    print(f"\ngbm vs logistic (log loss gain): {gain:+.5f}", flush=True)

    print("\ngbm calibration (tree ensembles often miscalibrate):", flush=True)
    print(calibration_table(ev["home_win"], ev["p_gbm"]).to_string(), flush=True)


if __name__ == "__main__":
    main()