"""Download one season of Retrosheet game logs and save as parquet."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"

# Retrosheet game logs ship without a header row. These are the
# 0-indexed positions of the fields we care about for a forecasting spine.
FIELDS = {
    0: "date",
    1: "game_num",
    2: "day_of_week",
    3: "vis_team",
    4: "vis_league",
    5: "vis_game_num",
    6: "home_team",
    7: "home_league",
    8: "home_game_num",
    9: "vis_score",
    10: "home_score",
    11: "length_outs",
    12: "day_night",
    16: "park_id",
    17: "attendance",
    18: "duration_min",
    101: "vis_sp_id",
    102: "vis_sp_name",
    103: "home_sp_id",
    104: "home_sp_name",
}


def fetch_season(year: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    print(f"downloading {url}", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        print(f"  extracting {name}", flush=True)
        with z.open(name) as f:
            raw = pd.read_csv(f, header=None, low_memory=False)

    print(f"  raw shape: {raw.shape}", flush=True)

    df = raw[list(FIELDS.keys())].copy()
    df.columns = list(FIELDS.values())
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["home_win"] = (df["home_score"] > df["vis_score"]).astype(int)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"gamelogs_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved: {out}", flush=True)
    return df


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    df = fetch_season(year)

    print(f"\nshape: {df.shape}", flush=True)
    print(f"home win rate: {df['home_win'].mean():.4f}", flush=True)
    print("\nfirst 3 rows:", flush=True)
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()