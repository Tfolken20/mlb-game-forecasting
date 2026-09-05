"""Join market odds onto the feature table, unioning two independent sources.

SOURCES
    2015-2020   Excel archive (odds.parquet)      -- one aggregated line
    2021-2024   JSON dataset  (odds_json.parquet) -- median across ~6 books

2021 exists in both. Cross-source validation on that overlap
(src/compare_odds_sources.py) found a de-vigged probability correlation of
0.9975, with 98.6% of games agreeing within 2 probability points. The JSON
source is preferred where both exist: it is a median across multiple books
rather than a single aggregated line, and it carries per-book detail.

VALIDATION
Every matched game is checked against an independently-sourced final score.
Doubleheaders are the known failure mode -- betting rotation order does not
reliably follow Retrosheet's scheduled game sequence -- so both orderings are
emitted upstream and the join keeps whichever the score confirms.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.parquet"
ODDS_XLS = ROOT / "data" / "processed" / "odds.parquet"
ODDS_JSN = ROOT / "data" / "processed" / "odds_json.parquet"
OUT = ROOT / "data" / "processed" / "modeling.parquet"

XLS_SEASONS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
JSN_SEASONS = [2021, 2022, 2023, 2024]

VALUE_COLS = [
    "p_market_home", "vig", "home_score_src", "vis_score_src",
    "n_books", "book_disagree", "odds_source",
]


def load_excel():
    df = pd.read_parquet(ODDS_XLS)
    df = df[df["season"].isin(XLS_SEASONS)].copy()
    df["odds_source"] = "excel"
    df["n_books"] = np.nan          # single aggregated line
    df["book_disagree"] = np.nan
    return df[["game_id", "game_id_alt"] + VALUE_COLS]


def load_json():
    df = pd.read_parquet(ODDS_JSN)
    df = df[df["season"].isin(JSN_SEASONS)].copy()
    df["odds_source"] = "json"
    return df[["game_id", "game_id_alt"] + VALUE_COLS]


def main():
    feat = pd.read_parquet(FEATURES)
    odds = pd.concat([load_excel(), load_json()], ignore_index=True)

    print("odds rows by source:", flush=True)
    print(odds["odds_source"].value_counts().to_string(), flush=True)

    dupes = odds["game_id"].duplicated().sum()
    if dupes:
        print(f"\nWARNING: {dupes} duplicate game_ids across sources", flush=True)
        odds = odds.drop_duplicates("game_id", keep="last")

    primary = odds[["game_id"] + VALUE_COLS]
    df = feat.merge(primary, on="game_id", how="left", validate="one_to_one")

    # Retry failures with the alternate doubleheader numbering.
    alt = (odds[["game_id_alt"] + VALUE_COLS]
           .rename(columns={"game_id_alt": "game_id"})
           .drop_duplicates("game_id"))

    bad = (
        df["p_market_home"].notna()
        & ((df["home_score"] != df["home_score_src"])
           | (df["vis_score"] != df["vis_score_src"]))
    )
    print(f"\nrows failing score check on primary join: {bad.sum()}", flush=True)
    if bad.any():
        fixed = df.loc[bad, ["game_id"]].merge(alt, on="game_id", how="left")
        for col in VALUE_COLS:
            df.loc[bad, col] = fixed[col].values

    covered = XLS_SEASONS + JSN_SEASONS
    ev = df[df["season"].isin(covered)]
    matched = ev["p_market_home"].notna()

    print(f"\ngames in covered seasons: {len(ev):,}", flush=True)
    print(f"matched to odds:          {matched.sum():,}  "
          f"({matched.mean():.2%})", flush=True)

    print("\nmatch rate by season:", flush=True)
    by_season = ev.groupby("season").agg(
        games=("game_id", "size"),
        matched=("p_market_home", lambda s: s.notna().sum()),
    )
    by_season["rate"] = (by_season["matched"] / by_season["games"]).round(4)
    print(by_season.to_string(), flush=True)

    both = ev[matched]
    score_ok = (
        (both["home_score"] == both["home_score_src"])
        & (both["vis_score"] == both["vis_score_src"])
    )
    print(f"\nscore agreement on matched games: {score_ok.mean():.4%}", flush=True)
    print(f"  disagreements: {(~score_ok).sum()}", flush=True)

    print("\nmarket calibration check:", flush=True)
    print(f"  mean implied home win prob: {both['p_market_home'].mean():.4f}", flush=True)
    print(f"  actual home win rate:       {both['home_win'].mean():.4f}", flush=True)

    print("\nby source:", flush=True)
    print(both.groupby("odds_source").agg(
        games=("game_id", "size"),
        implied=("p_market_home", "mean"),
        actual=("home_win", "mean"),
        vig=("vig", "median"),
    ).round(4).to_string(), flush=True)

    df.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()