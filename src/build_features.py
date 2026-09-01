"""Point-in-time features. Every value is computed before the game it describes."""
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPINE = ROOT / "data" / "processed" / "spine.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

ELO_K = 4.0
ELO_HOME_EDGE = 24.0
ELO_CARRYOVER = 0.75
ELO_START = 1500.0

SP_WINDOW = 8        # starts
TEAM_WINDOW = 20     # games
REST_CAP = 10        # days; beyond this is offseason or injury, not rotation rest
ESTABLISHED_STARTS = 5


def elo_expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def mean_or_nan(dq):
    return sum(dq) / len(dq) if dq else float("nan")


def build(df: pd.DataFrame) -> pd.DataFrame:
    elo = {}
    sp_runs = defaultdict(lambda: deque(maxlen=SP_WINDOW))   # runs allowed in their starts
    sp_last_date = {}
    sp_starts = defaultdict(int)
    team_rs = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))
    team_ra = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))

    rows = []
    prev_season = None

    for r in df.itertuples(index=False):
        if prev_season is not None and r.season != prev_season:
            for t in elo:
                elo[t] = ELO_START + ELO_CARRYOVER * (elo[t] - ELO_START)
        prev_season = r.season

        h_elo = elo.setdefault(r.home_team, ELO_START)
        v_elo = elo.setdefault(r.vis_team, ELO_START)

        rows.append({
            "game_id": r.game_id,
            "p_elo": elo_expected(h_elo + ELO_HOME_EDGE, v_elo),
            "elo_diff": (h_elo + ELO_HOME_EDGE) - v_elo,
            "home_sp_runs_avg": mean_or_nan(sp_runs[r.home_sp_id]),
            "vis_sp_runs_avg": mean_or_nan(sp_runs[r.vis_sp_id]),
            "home_sp_rest": (r.date - sp_last_date[r.home_sp_id]).days
                            if r.home_sp_id in sp_last_date else float("nan"),
            "vis_sp_rest": (r.date - sp_last_date[r.vis_sp_id]).days
                           if r.vis_sp_id in sp_last_date else float("nan"),
            # Career start counts rise monotonically with the calendar, so they are a
            # date proxy and must not be modeled directly. Saturating flag only:
            # does this pitcher have enough history for the rolling average to mean
            # anything yet?
            "home_sp_established": int(sp_starts[r.home_sp_id] >= ESTABLISHED_STARTS),
            "vis_sp_established": int(sp_starts[r.vis_sp_id] >= ESTABLISHED_STARTS),
            "home_rs_avg": mean_or_nan(team_rs[r.home_team]),
            "home_ra_avg": mean_or_nan(team_ra[r.home_team]),
            "vis_rs_avg": mean_or_nan(team_rs[r.vis_team]),
            "vis_ra_avg": mean_or_nan(team_ra[r.vis_team]),
        })

        # --- everything below updates state AFTER the row is emitted ---
        result = float(r.home_win)
        delta = ELO_K * (result - elo_expected(h_elo + ELO_HOME_EDGE, v_elo))
        elo[r.home_team] = h_elo + delta
        elo[r.vis_team] = v_elo - delta

        sp_runs[r.home_sp_id].append(r.vis_score)
        sp_runs[r.vis_sp_id].append(r.home_score)
        sp_last_date[r.home_sp_id] = r.date
        sp_last_date[r.vis_sp_id] = r.date
        sp_starts[r.home_sp_id] += 1
        sp_starts[r.vis_sp_id] += 1

        team_rs[r.home_team].append(r.home_score)
        team_ra[r.home_team].append(r.vis_score)
        team_rs[r.vis_team].append(r.vis_score)
        team_ra[r.vis_team].append(r.home_score)

    feat = pd.DataFrame(rows)
    out = df.merge(feat, on="game_id", how="left", validate="one_to_one")

    # Lower starter runs-allowed is better, so this is oriented so that
    # a higher value favors the home team.
    out["sp_runs_diff"] = out["vis_sp_runs_avg"] - out["home_sp_runs_avg"]
    out["team_net_diff"] = (
        (out["home_rs_avg"] - out["home_ra_avg"])
        - (out["vis_rs_avg"] - out["vis_ra_avg"])
    )

    for side in ("home", "vis"):
        out[f"{side}_sp_rest_capped"] = out[f"{side}_sp_rest"].clip(upper=REST_CAP)
    out["sp_rest_diff"] = out["home_sp_rest_capped"] - out["vis_sp_rest_capped"]

    return out


def main():
    df = pd.read_parquet(SPINE).sort_values(["date", "game_id"]).reset_index(drop=True)
    out = build(df)

    cols = ["elo_diff", "sp_runs_diff", "team_net_diff",
            "sp_rest_diff", "home_sp_rest_capped", "vis_sp_rest_capped"]

    print(f"rows: {len(out)}\n", flush=True)
    print("feature summary:", flush=True)
    print(out[cols].describe().round(3).to_string(), flush=True)

    print("\nmissing rate by feature:", flush=True)
    print(out[cols].isna().mean().round(4).to_string(), flush=True)

    print("\ncorrelation with home_win:", flush=True)
    print(out[cols + ["home_win"]].corr()["home_win"].drop("home_win").round(4).to_string(), flush=True)

    print("\nestablished-starter rate:", flush=True)
    print(out[["home_sp_established", "vis_sp_established"]].mean().round(4).to_string(), flush=True)

    out.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()