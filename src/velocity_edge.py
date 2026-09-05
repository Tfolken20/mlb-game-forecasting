"""Pre-registered test: does the market underprice starters throwing below
their own established velocity baseline?

MECHANISM (stated before results were examined): velocity is publicly
observable but is not a headline number. Books price primarily off results —
ERA, record, recent run support. A pitcher whose fastball has dropped 2+ mph
against his own baseline is showing a low-noise physical signal of fatigue or
injury that has not yet appeared in his results. If the market is slow to
incorporate it, the team starting that pitcher should win LESS often than the
closing line implies.

PRIMARY TEST: threshold of -2.0 mph, specified in advance. Other thresholds
are reported as a sensitivity curve, not as separate hypotheses.

Velocity delta was measured in src/stabilization.py at split-half r = 0.978,
the most reliable metric available — so a 2 mph deviation is signal, not noise.

Exploration seasons only. 2021 remains sealed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
PITCHER = ROOT / "data" / "processed" / "pitcher_features.parquet"
STARTS = ROOT / "data" / "processed" / "pitcher_starts.parquet"

LOCKBOX = [2024]
PRIMARY_THRESHOLD = -2.0
SENSITIVITY = [-1.0, -1.5, -2.0, -2.5, -3.0]
N_BOOT = 5000
RNG = np.random.default_rng(23)


def bridge_game_ids():
    starts = pd.read_parquet(STARTS)[["game_pk", "date", "team", "is_home"]]
    homes = starts[starts["is_home"]].rename(columns={"team": "home_team"})
    aways = starts[~starts["is_home"]].rename(columns={"team": "vis_team"})
    b = homes[["game_pk", "date", "home_team"]].merge(
        aways[["game_pk", "vis_team"]], on="game_pk", how="inner")
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values(["date", "game_pk"]).reset_index(drop=True)
    k = ["date", "vis_team", "home_team"]
    b["n"] = b.groupby(k)["game_pk"].transform("size")
    b["seq"] = b.groupby(k).cumcount() + 1
    b["game_num"] = np.where(b["n"] > 1, b["seq"], 0)
    b["game_id"] = (b["date"].dt.strftime("%Y%m%d") + "_"
                    + b["vis_team"] + "_" + b["home_team"] + "_"
                    + b["game_num"].astype(str))
    return b[["game_pk", "game_id"]]


def boot_ci(actual, implied):
    """Bootstrap CI for (actual win rate - market implied). Negative = overpriced."""
    n = len(actual)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        diffs[i] = actual[idx].mean() - implied[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def build_affected_frame(df):
    """One row per (game, affected side), from the perspective of the team
    whose starter is throwing below baseline."""
    home = pd.DataFrame({
        "game_id": df["game_id"],
        "season": df["season"],
        "delta": df["home_sp_velo_delta"],
        "won": df["home_win"],
        "implied": df["p_market_home"],
        "side": "home",
    })
    vis = pd.DataFrame({
        "game_id": df["game_id"],
        "season": df["season"],
        "delta": df["vis_sp_velo_delta"],
        "won": 1 - df["home_win"],
        "implied": 1 - df["p_market_home"],
        "side": "vis",
    })
    return pd.concat([home, vis], ignore_index=True).dropna(subset=["delta", "implied"])


def report(sub, label):
    if len(sub) < 100:
        return {"subset": label, "n": len(sub), "note": "underpowered"}
    actual = sub["won"].to_numpy(dtype=float)
    implied = sub["implied"].to_numpy(dtype=float)
    lo, hi = boot_ci(actual, implied)
    return {
        "subset": label,
        "n": len(sub),
        "market_implied": round(implied.mean(), 4),
        "actual": round(actual.mean(), 4),
        "bias": round(actual.mean() - implied.mean(), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "significant": "yes" if (hi < 0 or lo > 0) else "no",
    }


def main():
    df = pd.read_parquet(MODELING)
    pf = pd.read_parquet(PITCHER).merge(bridge_game_ids(), on="game_pk", how="inner")
    df = df.merge(pf.drop(columns=["game_pk"]), on="game_id", how="left")
    df = df[df["p_market_home"].notna()]

    held = df[df["season"].isin(LOCKBOX)]
    df = df[~df["season"].isin(LOCKBOX)]
    print(f"LOCKBOX: {LOCKBOX} withheld ({len(held):,} games, not examined)", flush=True)

    aff = build_affected_frame(df)
    print(f"exploration: {aff['game_id'].nunique():,} games, "
          f"{len(aff):,} team-games with a velocity baseline", flush=True)
    print(f"seasons: {sorted(aff['season'].unique())}\n", flush=True)

    print("PRIMARY TEST (pre-registered threshold: "
          f"{PRIMARY_THRESHOLD} mph)\n", flush=True)
    primary = aff[aff["delta"] <= PRIMARY_THRESHOLD]
    baseline = aff[aff["delta"] > PRIMARY_THRESHOLD]
    rows = [
        report(primary, f"starter down >= {abs(PRIMARY_THRESHOLD)} mph"),
        report(baseline, "all others"),
        report(aff, "all team-games"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("\n\nSENSITIVITY (not separate hypotheses — one curve):\n", flush=True)
    srows = [report(aff[aff["delta"] <= t], f"delta <= {t}") for t in SENSITIVITY]
    print(pd.DataFrame(srows).to_string(index=False), flush=True)

    print("\n\nDECILES OF VELOCITY DELTA:\n", flush=True)
    aff = aff.copy()
    aff["decile"] = pd.qcut(aff["delta"], 10, labels=False, duplicates="drop")
    drows = []
    for d, g in aff.groupby("decile"):
        r = report(g, f"decile {d}")
        r["delta_range"] = f"{g['delta'].min():.2f} to {g['delta'].max():.2f}"
        drows.append(r)
    cols = ["subset", "delta_range", "n", "market_implied", "actual",
            "bias", "ci_low", "ci_high", "significant"]
    print(pd.DataFrame(drows)[cols].to_string(index=False), flush=True)

    print("\n'bias' is actual win rate minus market implied.", flush=True)
    print("Negative means the market OVERRATED that team — the hypothesis.", flush=True)
    print("Only ci_high < 0 counts as evidence. Deciles are exploratory only.", flush=True)


if __name__ == "__main__":
    main()