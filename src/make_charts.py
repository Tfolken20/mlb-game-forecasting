"""Produce the two figures embedded in the README.

1. calibration.png  -- market vs. model, predicted probability against actual
                       outcome rate. Shows both are calibrated and the market
                       is tighter.
2. velocity_decay.png -- the pre-registered velocity effect measured at three
                       sample sizes, with confidence intervals. Shows the
                       effect regressing toward zero as data accumulates.

Both save to reports/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from velocity_edge import bridge_game_ids

ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "data" / "processed" / "modeling.parquet"
PITCHER = ROOT / "data" / "processed" / "pitcher_features.parquet"
REPORTS = ROOT / "reports"

FEATURES = ["elo_diff", "team_net_diff", "home_sp_rest_capped",
            "vis_sp_rest_capped", "sp_k_rate_diff", "sp_bb_rate_diff",
            "sp_velo_delta_diff"]

INK = "#1a1a1a"
MARKET = "#c1440e"
MODEL = "#2b6cb0"
GRID = "#d9d9d9"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def walk_forward(df):
    seasons = sorted(df["season"].unique())
    out = []
    for i, s in enumerate(seasons):
        if i < 2:
            continue
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == s]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        pipe.fit(train[FEATURES], train["home_win"])
        t = test[["game_id", "home_win", "p_market_home"]].copy()
        t["p_model"] = pipe.predict_proba(test[FEATURES])[:, 1]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def bin_curve(y, p, edges):
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 150:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


def chart_calibration(ev):
    edges = np.arange(0.25, 0.81, 0.05)
    y = ev["home_win"].to_numpy(dtype=float)

    mx, my, mn = bin_curve(y, ev["p_market_home"].to_numpy(dtype=float), edges)
    ox, oy, on = bin_curve(y, ev["p_model"].to_numpy(dtype=float), edges)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    style(ax)

    lims = [0.25, 0.80]
    ax.plot(lims, lims, color=INK, linewidth=1, linestyle="--",
            alpha=0.5, label="perfect calibration", zorder=1)
    ax.plot(mx, my, marker="o", markersize=6, linewidth=2,
            color=MARKET, label="Market (closing line)", zorder=3)
    ax.plot(ox, oy, marker="s", markersize=6, linewidth=2,
            color=MODEL, label="This model", zorder=2)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Predicted home win probability", fontsize=10, color=INK)
    ax.set_ylabel("Actual home win rate", fontsize=10, color=INK)
    ax.set_title("Both are well calibrated — the market's edge is sharpness, not calibration",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.text(0.01, -0.10,
             f"{len(ev):,} games, 2012-2024. Both models are honest about their own "
             "uncertainty. The market wins on accuracy\n(58.1% vs 56.8%) because it "
             "makes more confident predictions that are more often right, not because "
             "its\nprobabilities are better behaved.",
             fontsize=8, color="#666666", ha="left")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "calibration.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}", flush=True)


def chart_velocity_decay():
    """Measured results from three successive expansions of odds coverage."""
    labels = ["6 seasons\n21,367 team-games",
              "8 seasons\n32,913 team-games",
              "13 seasons\n55,391 team-games"]
    bias = [-0.0202, -0.0114, -0.0036]
    lo = [-0.0599, -0.0393, -0.0212]
    hi = [0.0191, 0.0158, 0.0140]

    x = np.arange(len(bias))
    lower = [b - l for b, l in zip(bias, lo)]
    upper = [h - b for b, h in zip(bias, hi)]

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)
    style(ax)

    ax.axhline(0, color=INK, linewidth=1, alpha=0.6, zorder=1)
    ax.errorbar(x, bias, yerr=[lower, upper], fmt="o", markersize=9,
                color=MODEL, ecolor=MODEL, elinewidth=2, capsize=7,
                capthick=2, zorder=3)

    for xi, b in zip(x, bias):
        ax.annotate(f"{b:+.4f}", (xi, b), textcoords="offset points",
                    xytext=(14, -4), fontsize=9.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(-0.45, len(bias) - 0.35)
    ax.set_ylabel("Actual win rate minus market implied", fontsize=10, color=INK)
    ax.set_title("A promising effect regressing to zero as data accumulates",
                 fontsize=12.5, color=INK, pad=12, loc="left")

    fig.text(0.01, -0.06,
             "Pre-registered test: do teams starting a pitcher 2+ mph below his own "
             "velocity baseline\nunderperform the closing line? The effect shrank 82% "
             "as the sample tripled. Bars are 95%\nbootstrap intervals; all three "
             "contain zero. The hypothesis was rejected without opening the lockbox.",
             fontsize=8, color="#666666", ha="left")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "velocity_decay.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}", flush=True)


def main():
    df = pd.read_parquet(MODELING)
    pf = pd.read_parquet(PITCHER).merge(bridge_game_ids(), on="game_pk", how="inner")
    df = df.merge(pf.drop(columns=["game_pk"]), on="game_id", how="left")
    df = df[df["p_market_home"].notna()].sort_values(["date", "game_id"])
    print(f"games: {len(df):,}", flush=True)

    ev = walk_forward(df)
    print(f"evaluated: {len(ev):,}", flush=True)

    chart_calibration(ev)
    chart_velocity_decay()


if __name__ == "__main__":
    main()