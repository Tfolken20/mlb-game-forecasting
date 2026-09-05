"""Cross-source price validation on the 2021 overlap.

The Excel archive (one aggregated line) and the JSON dataset (median of six
books) cover 2021 independently. If they agree on de-vigged probability, both
are measuring the same market and either can serve as the benchmark.

PRICES ONLY. No outcomes, no model predictions, no win rates. The 2021 lockbox
is not touched by this comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLS = ROOT / "data" / "processed" / "odds.parquet"
JSN = ROOT / "data" / "processed" / "odds_json.parquet"


def main():
    a = pd.read_parquet(XLS)
    b = pd.read_parquet(JSN)

    a = a[a["season"] == 2021][["game_id", "p_market_home", "vig", "home_ml_close"]]
    b = b[b["season"] == 2021][["game_id", "p_market_home", "vig",
                                "home_ml_close", "n_books", "book_disagree"]]
    a = a.rename(columns=lambda c: c + "_xls" if c != "game_id" else c)
    b = b.rename(columns=lambda c: c + "_jsn" if c != "game_id" else c)

    print(f"2021 games — excel: {len(a):,}   json: {len(b):,}", flush=True)

    m = a.merge(b, on="game_id", how="inner")
    print(f"matched on game_id: {len(m):,}\n", flush=True)

    d = m["p_market_home_jsn"] - m["p_market_home_xls"]
    print("de-vigged home probability, json minus excel:", flush=True)
    print(f"  mean diff:   {d.mean():+.5f}", flush=True)
    print(f"  median diff: {d.median():+.5f}", flush=True)
    print(f"  std:         {d.std():.5f}", flush=True)
    print(f"  correlation: {m['p_market_home_xls'].corr(m['p_market_home_jsn']):.5f}",
          flush=True)

    print("\nabsolute difference percentiles:", flush=True)
    print(d.abs().quantile([0.5, 0.75, 0.9, 0.99]).round(5).to_string(), flush=True)

    within = [(0.01, "1 pt"), (0.02, "2 pts"), (0.05, "5 pts")]
    print("\nshare of games where the two sources agree:", flush=True)
    for t, label in within:
        print(f"  within {label}: {(d.abs() <= t).mean():.2%}", flush=True)

    print(f"\nmedian vig — excel: {m['vig_xls'].median():.4f}   "
          f"json: {m['vig_jsn'].median():.4f}", flush=True)

    big = m.loc[d.abs() > 0.05, ["game_id", "home_ml_close_xls", "home_ml_close_jsn",
                                 "p_market_home_xls", "p_market_home_jsn",
                                 "n_books_jsn", "book_disagree_jsn"]]
    print(f"\ngames differing by more than 5 points: {len(big)}", flush=True)
    if len(big):
        print(big.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()