"""Pre-registered test: does the market under-adjust for park run environment?

MECHANISM (stated before results were examined): park does not favor the home
team -- both sides bat in it. What a hitter's park changes is VARIANCE. A higher
run environment widens the distribution of run differential, which compresses
win probability toward .500. A team that "should" win 60% of the time in a
neutral park should win somewhat less than 60% in Coors Field, because more
scoring gives the weaker side more chances to come out ahead.

If the market does not fully adjust for this, its probabilities will be too
EXTREME in high-run parks and not extreme enough in pitcher's parks.

PRIMARY TEST -- interaction. Fit, walk-forward:

    logit(P(home win)) = a + b * market_logit + c * market_logit * park_centered

The mechanism predicts c < 0: the market's signal should be DISCOUNTED in
high-run parks. A coefficient indistinguishable from zero means the market
already prices the variance effect.

This is an interaction hypothesis, so it is tested as one. Bucketing by park
alone would miss it entirely: park is symmetric between the teams, so its
main effect on home win probability should be near zero by construction.

SECONDARY -- bucketed favorite bias, for readability. Among games where the
market makes one side a clear favorite, does that favorite underperform its
implied probability in high-run parks?

Exploration seasons only. 2024 is sealed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from velocity_edge import bridge_game_ids  # noqa: F401  (kept for parity of imports)

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
PARK = ROOT / "data" / "processed" / "park_features.parquet"

LOCKBOX = [2024]
FAVORITE_THRESHOLD = 0.55     # "clear favorite" for the secondary test
N_BOOT = 5000
RNG = np.random.default_rng(31)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot_ci_mean_diff(actual, implied):
    n = len(actual)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        out[i] = actual[idx].mean() - implied[idx].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = pd.read_parquet(MODELING)[
        ["game_id", "season", "date", "home_win", "p_market_home"]]
    park = pd.read_parquet(PARK)[["game_id", "park_id", "park_factor",
                                  "park_prior_games"]]
    df = df.merge(park, on="game_id", how="left")
    df = df[df["p_market_home"].notna() & df["park_factor"].notna()]

    held = df[df["season"].isin(LOCKBOX)]
    df = df[~df["season"].isin(LOCKBOX)].sort_values(["date", "game_id"])
    print(f"LOCKBOX: {LOCKBOX} withheld ({len(held):,} games, not examined)", flush=True)
    print(f"exploration: {len(df):,} games "
          f"({df['season'].min()}-{df['season'].max()})\n", flush=True)

    df = df.copy()
    df["mlogit"] = logit(df["p_market_home"])
    df["park_c"] = df["park_factor"] - 1.0
    df["inter"] = df["mlogit"] * df["park_c"]

    # ---------------- PRIMARY: walk-forward interaction ----------------
    print("PRIMARY TEST -- interaction coefficient by season", flush=True)
    print("Mechanism predicts a NEGATIVE interaction (market signal should", flush=True)
    print("be discounted in high-run parks).\n", flush=True)

    seasons = sorted(df["season"].unique())
    coefs = []
    for i, s in enumerate(seasons):
        if i < 3:
            continue
        train = df[df["season"] < s]
        m = LogisticRegression(C=1e6, max_iter=1000)   # essentially unpenalized
        m.fit(train[["mlogit", "inter"]], train["home_win"])
        b_market, c_inter = m.coef_[0]
        coefs.append({"season": s, "b_market": round(b_market, 4),
                      "c_interaction": round(c_inter, 4)})
    print(pd.DataFrame(coefs).to_string(index=False), flush=True)

    # Full-sample fit with a bootstrap interval on the interaction.
    X = df[["mlogit", "inter"]].to_numpy()
    y = df["home_win"].to_numpy()
    full = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    print(f"\nfull-sample fit ({len(df):,} games):", flush=True)
    print(f"  market_logit coefficient:      {full.coef_[0][0]:+.4f}", flush=True)
    print(f"  interaction coefficient:       {full.coef_[0][1]:+.4f}", flush=True)

    boots = np.empty(400)
    n = len(df)
    for i in range(400):
        idx = RNG.integers(0, n, n)
        boots[i] = LogisticRegression(C=1e6, max_iter=1000).fit(
            X[idx], y[idx]).coef_[0][1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  interaction 95% CI:            ({lo:+.4f}, {hi:+.4f})", flush=True)
    verdict = "SUPPORTED" if hi < 0 else ("opposite sign" if lo > 0 else "null")
    print(f"  verdict: {verdict}", flush=True)

    # ---------------- SECONDARY: bucketed favorite bias ----------------
    print("\n\nSECONDARY -- favorite performance vs implied, by park quintile", flush=True)
    print(f"(clear favorites only: implied >= {FAVORITE_THRESHOLD})\n", flush=True)

    fav = df[(df["p_market_home"] >= FAVORITE_THRESHOLD)
             | (df["p_market_home"] <= 1 - FAVORITE_THRESHOLD)].copy()
    fav["fav_is_home"] = fav["p_market_home"] >= FAVORITE_THRESHOLD
    fav["fav_implied"] = np.where(fav["fav_is_home"],
                                  fav["p_market_home"], 1 - fav["p_market_home"])
    fav["fav_won"] = np.where(fav["fav_is_home"],
                              fav["home_win"], 1 - fav["home_win"])
    fav["quintile"] = pd.qcut(fav["park_factor"], 5, labels=False, duplicates="drop")

    rows = []
    for q, g in fav.groupby("quintile"):
        a = g["fav_won"].to_numpy(dtype=float)
        p = g["fav_implied"].to_numpy(dtype=float)
        lo_q, hi_q = boot_ci_mean_diff(a, p)
        rows.append({
            "quintile": int(q),
            "park_range": f"{g['park_factor'].min():.3f}-{g['park_factor'].max():.3f}",
            "n": len(g),
            "implied": round(p.mean(), 4),
            "actual": round(a.mean(), 4),
            "bias": round(a.mean() - p.mean(), 4),
            "ci_low": round(lo_q, 4),
            "ci_high": round(hi_q, 4),
        })
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("\nMechanism predicts bias becoming more NEGATIVE as park factor rises", flush=True)
    print("(favorites should underperform in high-run parks).", flush=True)

    # Coors on its own -- the most extreme park in baseball.
    coors = fav[fav["park_id"] == "DEN02"]
    if len(coors) > 100:
        a = coors["fav_won"].to_numpy(dtype=float)
        p = coors["fav_implied"].to_numpy(dtype=float)
        lo_c, hi_c = boot_ci_mean_diff(a, p)
        print(f"\nCoors Field alone (n={len(coors):,}):", flush=True)
        print(f"  favorites implied {p.mean():.4f}, actual {a.mean():.4f}, "
              f"bias {a.mean() - p.mean():+.4f}  CI ({lo_c:+.4f}, {hi_c:+.4f})",
              flush=True)


if __name__ == "__main__":
    main()