"""Parse the JSON odds dataset into one row per game with a median closing price.

Covers 2021 onward. The Excel archive covers 2015-2020. Both are unioned in
join_odds.py, with 2021 available in both as a cross-source validation overlap.

CONSENSUS IS COMPUTED IN PROBABILITY SPACE, NOT IN AMERICAN ODDS.

American odds are discontinuous: no valid line exists strictly between -100 and
+100. Taking a median directly across books straddling that gap produces
arithmetic nonsense -- four books at [-110, -104, +100, +105] give a "median" of
-2, which is not a price. This surfaced as impossible -3.0 medians and 100,000-
cent spreads when the output was compared against the independent Excel archive.

Each book is therefore converted to implied probability, de-vigged individually,
and the median taken across books. Probability is continuous on [0, 1].

Games where the books disagree by more than MAX_BOOK_DISAGREE are dropped. A
consensus price requires an actual consensus: when four books span 40
probability points on the same game, at least some of them are stale or broken,
and the median across them is arbitrary rather than informative. This threshold
was set after cross-source comparison showed every large disagreement with the
Excel archive came from games with extreme book spread.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "odds" / "mlb_odds_dataset.json"
OUT = ROOT / "data" / "processed" / "odds_json.parquet"

MIN_BOOKS = 3
MAX_ABS_ODDS = 5000          # beyond this is a sentinel, not a price
MAX_BOOK_DISAGREE = 0.10     # probability points across books

TEAM_MAP = {
    "ARI": "ARI", "AZ": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CWS": "CHA", "CHW": "CHA",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KCA", "KCR": "KCA",
    "LAA": "ANA", "LAD": "LAN", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYN", "NYY": "NYA",
    "OAK": "OAK", "ATH": "OAK",
    "PHI": "PHI", "PIT": "PIT",
    "SD": "SDN", "SDP": "SDN", "SEA": "SEA",
    "SF": "SFN", "SFG": "SFN", "STL": "SLN",
    "TB": "TBA", "TBR": "TBA", "TEX": "TEX", "TOR": "TOR",
    "WSH": "WAS", "WAS": "WAS",
}


def american_to_prob(ml):
    ml = float(ml)
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def prob_to_american(p):
    """Inverse, for reporting a human-readable consensus line."""
    if p <= 0 or p >= 1:
        return np.nan
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def valid_pair(home, away):
    if home is None or away is None:
        return False
    return (100 <= abs(home) <= MAX_ABS_ODDS) and (100 <= abs(away) <= MAX_ABS_ODDS)


def devig(home, away):
    """One book's pair -> de-vigged home probability and that book's hold."""
    ph, pa = american_to_prob(home), american_to_prob(away)
    total = ph + pa
    return ph / total, total - 1.0


def main():
    print(f"loading {SRC.name}...", flush=True)
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped_type = 0
    skipped_books = 0
    skipped_disagree = 0
    dropped_prices = 0

    for date_str, games in data.items():
        for g in games:
            gv = g.get("gameView", {})
            if gv.get("gameType") != "R":
                skipped_type += 1
                continue

            ml = (g.get("odds", {}) or {}).get("moneyline") or []
            close_p, close_vig, open_p = [], [], []
            raw_home_close = []

            for e in ml:
                cl = e.get("currentLine") or {}
                ol = e.get("openingLine") or {}

                h, a = cl.get("homeOdds"), cl.get("awayOdds")
                if valid_pair(h, a):
                    p, v = devig(h, a)
                    close_p.append(p)
                    close_vig.append(v)
                    raw_home_close.append(float(h))
                elif h is not None or a is not None:
                    dropped_prices += 1

                oh, oa = ol.get("homeOdds"), ol.get("awayOdds")
                if valid_pair(oh, oa):
                    open_p.append(devig(oh, oa)[0])

            if len(close_p) < MIN_BOOKS:
                skipped_books += 1
                continue

            disagree = float(np.max(close_p) - np.min(close_p))
            if disagree > MAX_BOOK_DISAGREE:
                skipped_disagree += 1
                continue

            p_med = float(np.median(close_p))
            rows.append({
                "date": pd.Timestamp(date_str),
                "vis_team_src": (gv.get("awayTeam") or {}).get("shortName"),
                "home_team_src": (gv.get("homeTeam") or {}).get("shortName"),
                "vis_score_src": gv.get("awayTeamScore"),
                "home_score_src": gv.get("homeTeamScore"),
                "start": gv.get("startDate"),
                "n_books": len(close_p),
                "p_market_home": p_med,
                "p_market_home_open": float(np.median(open_p)) if open_p else np.nan,
                "vig": float(np.median(close_vig)),
                "book_disagree": disagree,
                "home_ml_best": float(np.max(raw_home_close)),
                "home_ml_close": prob_to_american(p_med),
            })

    df = pd.DataFrame(rows)
    print(f"kept {len(df):,} games", flush=True)
    print(f"  skipped {skipped_type:,} non-regular-season", flush=True)
    print(f"  skipped {skipped_books:,} with <{MIN_BOOKS} valid books", flush=True)
    print(f"  skipped {skipped_disagree:,} with book disagreement "
          f">{MAX_BOOK_DISAGREE}", flush=True)
    print(f"  dropped {dropped_prices:,} invalid individual prices", flush=True)

    all_codes = sorted(set(df["vis_team_src"].dropna()) | set(df["home_team_src"].dropna()))
    missing = [t for t in all_codes if t not in TEAM_MAP]
    if missing:
        print(f"\nUNMAPPED TEAM CODES: {missing}", flush=True)

    df["vis_team"] = df["vis_team_src"].map(TEAM_MAP)
    df["home_team"] = df["home_team_src"].map(TEAM_MAP)
    df["season"] = df["date"].dt.year

    df = df.sort_values(["date", "start"]).reset_index(drop=True)
    key = ["date", "vis_team", "home_team"]
    df["n_in_day"] = df.groupby(key)["start"].transform("size")
    df["seq"] = df.groupby(key).cumcount() + 1
    df["game_num"] = np.where(df["n_in_day"] > 1, df["seq"], 0)
    df["game_num_alt"] = np.where(df["n_in_day"] > 1, df["n_in_day"] - df["seq"] + 1, 0)

    def make_id(col):
        return (
            df["date"].dt.strftime("%Y%m%d")
            + "_" + df["vis_team"].astype(str)
            + "_" + df["home_team"].astype(str)
            + "_" + df[col].astype(str)
        )

    df["game_id"] = make_id("game_num")
    df["game_id_alt"] = make_id("game_num_alt")

    print("\ngames per season:", flush=True)
    print(df.groupby("season").size().to_string(), flush=True)
    print(f"\nduplicate game_ids: {df['game_id'].duplicated().sum()}", flush=True)
    print(f"median vig: {df['vig'].median():.4f}", flush=True)
    print(f"median books per game: {df['n_books'].median():.0f}", flush=True)

    print("\nconsensus home probability:", flush=True)
    print(df["p_market_home"].describe().round(4).to_string(), flush=True)
    print("\nbook disagreement (probability points):", flush=True)
    print(df["book_disagree"].describe().round(4).to_string(), flush=True)

    print(f"\nmarket home win prob (mean): {df['p_market_home'].mean():.4f}", flush=True)
    print(f"actual home win rate:        "
          f"{(df['home_score_src'] > df['vis_score_src']).mean():.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()