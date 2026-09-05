"""Pull one season of Statcast pitch-level data, cached by chunk so it can resume."""
import sys
import time
from pathlib import Path

import pandas as pd
from pybaseball import cache

cache.enable()   # day-level cache; survives chunk failures and re-pulls

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "statcast"

# Half-month chunks: a failure costs less, and each caches independently.
CHUNKS = [(3, 20, 3, 31),
          (4, 1, 4, 15), (4, 16, 4, 30),
          (5, 1, 5, 15), (5, 16, 5, 31),
          (6, 1, 6, 15), (6, 16, 6, 30),
          (7, 1, 7, 15), (7, 16, 7, 31),
          (8, 1, 8, 15), (8, 16, 8, 31),
          (9, 1, 9, 15), (9, 16, 9, 30),
          (10, 1, 10, 15)]

KEEP = [
    "game_date", "game_pk", "game_year", "game_type",
    "pitcher", "player_name", "batter",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "home_team", "away_team",
    "pitch_type", "release_speed", "release_spin_rate",
    "events", "description", "type",
    "balls", "strikes", "outs_when_up",
    "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "delta_run_exp",
]


def fetch_chunk(year, m1, d1, m2, d2):
    from pybaseball import statcast
    start = f"{year}-{m1:02d}-{d1:02d}"
    end = f"{year}-{m2:02d}-{d2:02d}"
    df = statcast(start_dt=start, end_dt=end, verbose=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Regular season only. Spring training stats are near-meaningless and would
    # corrupt any rolling pitcher average; postseason is priced differently.
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in KEEP if c in df.columns]
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        print(f"    note: columns absent from source: {missing}", flush=True)
    return df[cols]


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    CACHE.mkdir(parents=True, exist_ok=True)

    parts = []
    for (m1, d1, m2, d2) in CHUNKS:
        label = f"{year}-{m1:02d}-{d1:02d}"
        out = CACHE / f"statcast_{year}_{m1:02d}_{d1:02d}.parquet"

        if out.exists():
            print(f"  {label}: cached", flush=True)
            parts.append(pd.read_parquet(out))
            continue

        print(f"  {label}: fetching...", flush=True)
        t0 = time.time()
        try:
            df = fetch_chunk(year, m1, d1, m2, d2)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            print("    rerun to resume; completed chunks are cached", flush=True)
            continue

        if df.empty:
            print("    no regular-season data in range", flush=True)
            continue

        df.to_parquet(out, index=False)
        print(f"    {len(df):,} pitches in {time.time() - t0:.0f}s", flush=True)
        parts.append(df)

    if not parts:
        raise SystemExit("Nothing fetched.")

    season = pd.concat(parts, ignore_index=True)
    print(f"\ntotal pitches:   {len(season):,}", flush=True)
    print(f"unique games:    {season['game_pk'].nunique():,}", flush=True)
    print(f"unique pitchers: {season['pitcher'].nunique():,}", flush=True)

    if "game_type" in season.columns:
        print(f"\ngame types present: "
              f"{season['game_type'].value_counts().to_dict()}", flush=True)

    print("\nsample rows:", flush=True)
    show = [c for c in ["game_date", "game_pk", "player_name", "inning",
                        "pitch_type", "release_speed", "events", "delta_run_exp"]
            if c in season.columns]
    print(season[show].head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()