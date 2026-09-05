"""Pre-registered subset analysis. Exploration seasons only; 2021 is held out.

Each subset below was specified with a stated mechanism BEFORE any results were
examined. Subsets without a prior reason to expect mispricing are data mining,
not hypotheses.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"

LOCKBOX_SEASONS = [2021]        # do not touch until a hypothesis is final
FEATURES = ["elo_diff", "sp_runs_diff", "team_net_diff",
            "home_sp_rest_capped", "vis_sp_rest_capped"]
N_BOOT = 2000
RNG = np.random.default_rng(17)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk_forward(df):
    seasons = sorted(df["season"].unique())
    out = []
    for season in seasons[2:]:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        pipe.fit(train[FEATURES], train["home_win"])
        t = test.copy()
        t["p_own"] = pipe.predict_proba(test[FEATURES])[:, 1]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def boot_diff(y, p_market, p_own):
    """Bootstrap CI for (market log loss - own log loss). Positive favors us."""
    y = np.asarray(y); pm = np.asarray(p_market); po = np.asarray(p_own)
    n = len(y)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        diffs[i] = log_loss(y[idx], pm[idx]) - log_loss(y[idx], po[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def define_subsets(df):
    """Each entry: (name, mask, stated mechanism)."""
    game_no = df.groupby(["season", "home_team"]).cumcount()
    disagree = (df["p_own"] - df["p_market_home"]).abs()
    return [
        ("all games", pd.Series(True, index=df.index),
         "reference"),
        ("first 15 games of season", game_no < 15,
         "market has little current-season information"),
        ("unproven starter", (df["home_sp_established"] == 0) | (df["vis_sp_established"] == 0),
         "pitchers without track record are hard to price"),
        ("doubleheader game 2", df["game_num"] == 2,
         "bullpen depletion from game 1 is slow to price"),
        ("short rest starter", (df["home_sp_rest_capped"] <= 4) | (df["vis_sp_rest_capped"] <= 4),
         "non-standard rotation slot, often a late decision"),
        ("widest vig quintile", df["vig"] >= df["vig"].quantile(0.8),
         "wide markets signal low liquidity and low book confidence"),
        ("largest disagreement decile", disagree >= disagree.quantile(0.9),
         "where our information differs most from consensus"),
    ]


def main():
    df = pd.read_parquet(MODELING)
    df = df[df["p_market_home"].notna()]

    held = df[df["season"].isin(LOCKBOX_SEASONS)]
    df = df[~df["season"].isin(LOCKBOX_SEASONS)]
    print(f"LOCKBOX: {LOCKBOX_SEASONS} withheld ({len(held)} games, not examined)", flush=True)

    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    ev = walk_forward(df)
    print(f"exploration set: {len(ev)} games "
          f"({ev['season'].min()}-{ev['season'].max()})\n", flush=True)

    rows = []
    for name, mask, mechanism in define_subsets(ev):
        sub = ev[mask.reindex(ev.index, fill_value=False)]
        if len(sub) < 200:
            rows.append({"subset": name, "n": len(sub), "note": "too small to test"})
            continue
        ll_m = log_loss(sub["home_win"], sub["p_market_home"])
        ll_o = log_loss(sub["home_win"], sub["p_own"])
        lo, hi = boot_diff(sub["home_win"], sub["p_market_home"], sub["p_own"])
        rows.append({
            "subset": name,
            "n": len(sub),
            "market_ll": round(ll_m, 5),
            "own_ll": round(ll_o, 5),
            "gain": round(ll_m - ll_o, 5),
            "ci_low": round(lo, 5),
            "ci_high": round(hi, 5),
            "beats_market": "yes" if lo > 0 else "no",
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False), flush=True)

    print("\nmechanisms (pre-specified):", flush=True)
    for name, _, mech in define_subsets(ev):
        print(f"  {name:32s} {mech}", flush=True)

    print("\nNote: 'gain' positive means our model beat the market on that subset.", flush=True)
    print("Only treat a subset as real if ci_low > 0, and even then expect", flush=True)
    print("regression — 7 subsets tested means roughly a 1-in-3 chance of one", flush=True)
    print("false positive at 95% confidence.", flush=True)


if __name__ == "__main__":
    main()