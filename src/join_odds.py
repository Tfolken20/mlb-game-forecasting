"""Join market odds onto the feature table and validate the match against scores."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.parquet"
ODDS = ROOT / "data" / "processed" / "odds.parquet"
OUT = ROOT / "data" / "processed" / "modeling.parquet"

ODDS_COLS = [
    "game_id", "p_market_home", "vig",
    "home_ml_close", "vis_ml_close", "home_ml_open", "vis_ml_open",
    "home_score_src", "vis_score_src",
    "home_pitcher_src", "vis_pitcher_src",
]
VALUE_COLS = [c for c in ODDS_COLS if c != "game_id"]


def main():
    feat = pd.read_parquet(FEATURES)
    odds_full = pd.read_parquet(ODDS)
    odds = odds_full[ODDS_COLS]

    covered = sorted(set(feat["season"]) & set(odds_full["season"]))
    print(f"seasons with odds: {covered}", flush=True)

    df = feat.merge(odds, on="game_id", how="left", validate="one_to_one")

    # Where the primary join disagrees on score, retry with the alternate
    # doubleheader numbering and keep the version the scores confirm.
    alt = odds_full[["game_id_alt"] + VALUE_COLS].rename(
        columns={"game_id_alt": "game_id"}
    ).drop_duplicates("game_id")

    bad = (
        df["p_market_home"].notna()
        & ((df["home_score"] != df["home_score_src"])
           | (df["vis_score"] != df["vis_score_src"]))
    )
    print(f"rows failing score check on primary join: {bad.sum()}", flush=True)

    if bad.any():
        fixed = df.loc[bad, ["game_id"]].merge(alt, on="game_id", how="left")
        for col in VALUE_COLS:
            df.loc[bad, col] = fixed[col].values

    ev = df[df["season"].isin(covered)]
    matched = ev["p_market_home"].notna()
    print(f"\ngames in covered seasons: {len(ev)}", flush=True)
    print(f"matched to odds:          {matched.sum()}  ({matched.mean():.4%})", flush=True)

    print("\nmatch rate by season:", flush=True)
    print(ev.groupby("season")["p_market_home"]
            .apply(lambda s: round(s.notna().mean(), 4)).to_string(), flush=True)

    # Independent validation: scores from two unrelated sources must agree.
    both = ev[matched]
    score_ok = (
        (both["home_score"] == both["home_score_src"])
        & (both["vis_score"] == both["vis_score_src"])
    )
    print(f"\nscore agreement on matched games: {score_ok.mean():.4%}", flush=True)
    if not score_ok.all():
        print(f"\nremaining disagreements ({(~score_ok).sum()}):", flush=True)
        cols = ["game_id", "home_team", "vis_team", "home_score",
                "home_score_src", "vis_score", "vis_score_src"]
        print(both.loc[~score_ok, cols].head(10).to_string(index=False), flush=True)

    unmatched = ev[~matched]
    if len(unmatched):
        print(f"\nsample of unmatched games ({len(unmatched)}):", flush=True)
        print(unmatched[["game_id", "date", "vis_team", "home_team"]]
              .head(10).to_string(index=False), flush=True)

    df.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()