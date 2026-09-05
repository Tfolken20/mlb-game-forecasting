"""Do measured-reliability pitcher features beat the market, or the old proxy?"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from metrics import summarize, log_loss

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
PITCHER = ROOT / "data" / "processed" / "pitcher_features.parquet"
STARTS = ROOT / "data" / "processed" / "pitcher_starts.parquet"

OLD = ["elo_diff", "sp_runs_diff", "team_net_diff",
       "home_sp_rest_capped", "vis_sp_rest_capped"]
NEW = ["elo_diff", "team_net_diff", "home_sp_rest_capped", "vis_sp_rest_capped",
       "sp_k_rate_diff", "sp_bb_rate_diff", "sp_xwoba_diff", "sp_velo_delta_diff"]


def walk_forward(df, cols, label):
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
        pipe.fit(train[cols], train["home_win"])
        out.append(pd.DataFrame({
            "game_id": test["game_id"].values,
            label: pipe.predict_proba(test[cols])[:, 1],
        }))
    return pd.concat(out, ignore_index=True)


def main():
    df = pd.read_parquet(MODELING)

    # Bridge Statcast game_pk to the Retrosheet game_id via date + teams.
    starts = pd.read_parquet(STARTS)[["game_pk", "date", "team", "is_home"]]
    homes = starts[starts["is_home"]].rename(columns={"team": "home_team"})
    aways = starts[~starts["is_home"]].rename(columns={"team": "vis_team"})
    bridge = homes[["game_pk", "date", "home_team"]].merge(
        aways[["game_pk", "vis_team"]], on="game_pk", how="inner")
    bridge["date"] = pd.to_datetime(bridge["date"])

    # Doubleheaders: number within date+teams the same way the spine does.
    bridge = bridge.sort_values(["date", "game_pk"]).reset_index(drop=True)
    k = ["date", "vis_team", "home_team"]
    bridge["n"] = bridge.groupby(k)["game_pk"].transform("size")
    bridge["seq"] = bridge.groupby(k).cumcount() + 1
    bridge["game_num"] = np.where(bridge["n"] > 1, bridge["seq"], 0)
    bridge["game_id"] = (bridge["date"].dt.strftime("%Y%m%d") + "_"
                         + bridge["vis_team"] + "_" + bridge["home_team"] + "_"
                         + bridge["game_num"].astype(str))

    pf = pd.read_parquet(PITCHER).merge(
        bridge[["game_pk", "game_id"]], on="game_pk", how="inner")
    print(f"pitcher features bridged to game_id: {len(pf):,}", flush=True)

    df = df.merge(pf.drop(columns=["game_pk"]), on="game_id", how="left")
    match = df["sp_k_rate_diff"].notna().mean()
    print(f"games with pitcher features: {match:.2%}", flush=True)

    ev = df[df["p_market_home"].notna()].sort_values(["date", "game_id"]).reset_index(drop=True)
    print(f"games with market price: {len(ev):,}\n", flush=True)

    old = walk_forward(ev, OLD, "p_old")
    new = walk_forward(ev, NEW, "p_new")
    res = ev.merge(old, on="game_id").merge(new, on="game_id")
    print(f"evaluated: {len(res):,} games "
          f"({res['season'].min()}-{res['season'].max()})\n", flush=True)

    rows = [
        summarize(res["home_win"], res["p_elo"], "elo"),
        summarize(res["home_win"], res["p_old"], "old features (runs proxy)"),
        summarize(res["home_win"], res["p_new"], "new features (statcast)"),
        summarize(res["home_win"], res["p_market_home"], "market (closing)"),
    ]
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    gain = log_loss(res["home_win"], res["p_old"]) - log_loss(res["home_win"], res["p_new"])
    gap = log_loss(res["home_win"], res["p_new"]) - log_loss(res["home_win"], res["p_market_home"])
    print(f"\nnew vs old features (log loss gain): {gain:+.5f}", flush=True)
    print(f"new features vs market (gap):        {gap:+.5f}  "
          f"(positive means market still ahead)", flush=True)

    print("\nlog loss by season:", flush=True)
    by = res.groupby("season").apply(lambda g: pd.Series({
        "old": round(log_loss(g["home_win"], g["p_old"]), 5),
        "new": round(log_loss(g["home_win"], g["p_new"]), 5),
        "market": round(log_loss(g["home_win"], g["p_market_home"]), 5),
    }), include_groups=False)
    print(by.to_string(), flush=True)


if __name__ == "__main__":
    main()