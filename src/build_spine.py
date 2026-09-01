"""Combine per-season game logs into a single sorted spine table."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main():
    files = sorted(RAW.glob("gamelogs_*.parquet"))
    if not files:
        raise SystemExit("No game log files found. Run fetch_gamelogs.py first.")

    print(f"found {len(files)} season files", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Unique key: doubleheaders share a date, so game_num disambiguates.
    df["game_id"] = (
        df["date"].dt.strftime("%Y%m%d")
        + "_" + df["vis_team"]
        + "_" + df["home_team"]
        + "_" + df["game_num"].astype(str)
    )
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    df["season"] = df["date"].dt.year

    dupes = df["game_id"].duplicated().sum()
    print(f"rows: {len(df)}  duplicate game_ids: {dupes}", flush=True)
    print("\ngames and home win rate by season:", flush=True)
    print(df.groupby("season").agg(
        games=("game_id", "size"),
        home_win_rate=("home_win", "mean"),
    ).round(4).to_string())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "spine.parquet"
    df.to_parquet(out, index=False)
    print(f"\nsaved: {out}", flush=True)


if __name__ == "__main__":
    main()