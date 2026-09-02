"""Inspect a downloaded odds file before writing any parser against it."""
import sys
from pathlib import Path

import pandas as pd

ODDS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "odds"


def main():
    files = sorted(list(ODDS_DIR.glob("*.xlsx")) + list(ODDS_DIR.glob("*.xls"))
                   + list(ODDS_DIR.glob("*.csv")))
    if not files:
        raise SystemExit(f"No files found in {ODDS_DIR}")

    print("files found:", flush=True)
    for f in files:
        print(f"  {f.name}")

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else files[0]
    print(f"\ninspecting: {target.name}\n", flush=True)

    if target.suffix.lower() == ".csv":
        df = pd.read_csv(target)
    else:
        df = pd.read_excel(target)

    print(f"shape: {df.shape}\n", flush=True)
    print("columns:", flush=True)
    for c in df.columns:
        print(f"  {c!r}  ({df[c].dtype})")

    print("\nfirst 6 rows:", flush=True)
    print(df.head(6).to_string(), flush=True)

    for col in df.columns:
        if str(col).strip().lower() in ("vh", "v/h", "team"):
            print(f"\nvalue counts for {col!r}:", flush=True)
            print(df[col].value_counts().head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()