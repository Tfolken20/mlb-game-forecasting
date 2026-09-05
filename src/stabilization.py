"""How many starts before a pitcher metric means anything?

Split-half reliability: for pitchers with at least 2k starts in a season,
randomly split k starts into each half and correlate the metric across halves.
Sweeping k shows how fast each metric stabilizes. A metric that needs 20 starts
to become reliable is nearly useless for in-season prediction no matter how
well it describes the past.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STARTS = ROOT / "data" / "processed" / "pitcher_starts.parquet"

K_VALUES = [2, 3, 5, 8, 12, 16, 20]
N_REPEATS = 30
RNG = np.random.default_rng(11)

# metric -> (numerator column, denominator column)
# Rate metrics are recomputed from sums so each half is properly weighted,
# not averaged over averages.
METRICS = {
    "k_rate":         ("strikeouts",   "batters_faced"),
    "bb_rate":        ("walks",        "batters_faced"),
    "run_exp_per_bf": ("run_exp_total", "batters_faced"),
    "hr_rate":        ("home_runs",    "batters_faced"),
    "xwoba_mean":     ("xwoba_weighted", "batters_faced"),
    "velo_mean":      ("velo_weighted",  "pitches"),
}


def split_half_r(df, num_col, den_col, k):
    """Correlate the metric between two random k-start halves, per pitcher-season."""
    rs = []
    for _ in range(N_REPEATS):
        a_vals, b_vals = [], []
        for _, g in df.groupby(["starter_id", "season"], sort=False):
            if len(g) < 2 * k:
                continue
            idx = RNG.permutation(len(g))
            a = g.iloc[idx[:k]]
            b = g.iloc[idx[k:2 * k]]
            da, db = a[den_col].sum(), b[den_col].sum()
            if da == 0 or db == 0:
                continue
            a_vals.append(a[num_col].sum() / da)
            b_vals.append(b[num_col].sum() / db)
        if len(a_vals) > 10:
            rs.append(np.corrcoef(a_vals, b_vals)[0, 1])
    if not rs:
        return np.nan, 0
    return float(np.mean(rs)), len(a_vals)


def main():
    df = pd.read_parquet(STARTS)
    df["season"] = pd.to_datetime(df["date"]).dt.year

    # Weight the mean-based columns so halves can be recombined from sums.
    df["xwoba_weighted"] = df["xwoba_mean"] * df["batters_faced"]
    df["velo_weighted"] = df["velo_mean"] * df["pitches"]

    print(f"starts: {len(df):,}", flush=True)
    print(f"pitcher-seasons: {df.groupby(['starter_id', 'season']).ngroups:,}\n", flush=True)

    rows = []
    for name, (num, den) in METRICS.items():
        sub = df[df[num].notna() & df[den].notna()]
        row = {"metric": name}
        for k in K_VALUES:
            r, n = split_half_r(sub, num, den, k)
            row[f"k={k}"] = round(r, 3) if not np.isnan(r) else None
            row[f"n{k}"] = n
        rows.append(row)
        print(f"  {name} done", flush=True)

    res = pd.DataFrame(rows)
    show = ["metric"] + [f"k={k}" for k in K_VALUES]
    print("\nsplit-half correlation by number of starts per half:", flush=True)
    print(res[show].to_string(index=False), flush=True)

    print("\npitcher-seasons available at each k:", flush=True)
    print(res[["metric"] + [f"n{k}" for k in K_VALUES]].to_string(index=False), flush=True)

    print("\nReading this: r ~0.5 is the conventional stabilization threshold.", flush=True)
    print("A metric still near 0.2 at k=12 carries little signal within a season.", flush=True)


if __name__ == "__main__":
    main()