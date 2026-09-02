"""Parse SBR odds workbooks into one row per game and map to Retrosheet team codes."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = ROOT / "data" / "raw" / "odds"
OUT = ROOT / "data" / "processed" / "odds.parquet"

# SBR abbreviation -> Retrosheet code. Aliases included where the source
# has used more than one form across seasons.
TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL",
    "BOS": "BOS", "BRS": "BOS",
    "CUB": "CHN", "CHC": "CHN",
    "CWS": "CHA", "CHW": "CHA", "CHI": "CHA",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU",
    "KAN": "KCA", "KC": "KCA", "KCR": "KCA",
    "LAA": "ANA", "ANA": "ANA",
    "LOS": "LAN", "LAD": "LAN",
    "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL", "MIN": "MIN",
    "NYM": "NYN", "NYY": "NYA",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SDG": "SDN", "SD": "SDN", "SDP": "SDN",
    "SEA": "SEA",
    "SFO": "SFN", "SF": "SFN", "SFG": "SFN",
    "STL": "SLN",
    "TAM": "TBA", "TB": "TBA", "TBR": "TBA",
    "TEX": "TEX", "TOR": "TOR",
    "WAS": "WAS", "WSH": "WAS",
}


def parse_date(raw, year: int):
    """405 -> Apr 5; 1005 -> Oct 5."""
    s = str(int(raw))
    if len(s) == 3:
        month, day = int(s[0]), int(s[1:])
    elif len(s) == 4:
        month, day = int(s[:2]), int(s[2:])
    else:
        return pd.NaT
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def american_to_prob(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def load_season(path: Path) -> pd.DataFrame:
    year = int(re.search(r"(\d{4})", path.stem).group(1))
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    vh = raw["VH"].astype(str).str.strip().str.upper()
    vis = raw[vh == "V"].reset_index(drop=True)
    home = raw[vh == "H"].reset_index(drop=True)

    if len(vis) != len(home):
        raise SystemExit(f"{path.name}: {len(vis)} visitor rows vs {len(home)} home rows")

    df = pd.DataFrame({
        "date": [parse_date(d, year) for d in vis["Date"]],
        "vis_team_src": vis["Team"].astype(str).str.strip().str.upper(),
        "home_team_src": home["Team"].astype(str).str.strip().str.upper(),
        "vis_pitcher_src": vis["Pitcher"].astype(str).str.strip(),
        "home_pitcher_src": home["Pitcher"].astype(str).str.strip(),
        "vis_score_src": pd.to_numeric(vis["Final"], errors="coerce"),
        "home_score_src": pd.to_numeric(home["Final"], errors="coerce"),
        "vis_ml_open": pd.to_numeric(vis["Open"], errors="coerce"),
        "home_ml_open": pd.to_numeric(home["Open"], errors="coerce"),
        "vis_ml_close": pd.to_numeric(vis["Close"], errors="coerce"),
        "home_ml_close": pd.to_numeric(home["Close"], errors="coerce"),
        "rot": pd.to_numeric(vis["Rot"], errors="coerce"),
    })
    df["season"] = year
    return df


def main():
    files = sorted(ODDS_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files in {ODDS_DIR}")

    frames = []
    for f in files:
        d = load_season(f)
        print(f"{f.name}: {len(d)} games", flush=True)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # --- team code mapping ---
    all_codes = sorted(set(df["vis_team_src"]) | set(df["home_team_src"]))
    unknown = [t for t in all_codes if t not in TEAM_MAP]
    if unknown:
        print(f"\nUNMAPPED TEAM CODES: {unknown}", flush=True)
        print("Add these to TEAM_MAP before continuing.\n", flush=True)

    df["vis_team"] = df["vis_team_src"].map(TEAM_MAP)
    df["home_team"] = df["home_team_src"].map(TEAM_MAP)

    # --- doubleheader numbering ---
    # Rotation number does not reliably follow Retrosheet's scheduled game
    # sequence, so both orderings are emitted. The join downstream keeps
    # whichever one agrees with the independently-sourced final score.
    df = df.sort_values(["date", "rot"]).reset_index(drop=True)
    key = ["date", "vis_team", "home_team"]
    df["n_in_day"] = df.groupby(key)["rot"].transform("size")
    df["seq"] = df.groupby(key).cumcount() + 1
    df["game_num"] = np.where(df["n_in_day"] > 1, df["seq"], 0)
    df["game_num_alt"] = np.where(df["n_in_day"] > 1, df["n_in_day"] - df["seq"] + 1, 0)

    def make_id(num_col):
        return (
            df["date"].dt.strftime("%Y%m%d")
            + "_" + df["vis_team"].astype(str)
            + "_" + df["home_team"].astype(str)
            + "_" + df[num_col].astype(str)
        )

    df["game_id"] = make_id("game_num")
    df["game_id_alt"] = make_id("game_num_alt")

    # --- de-vigged market probability ---
    p_vis_raw = american_to_prob(df["vis_ml_close"])
    p_home_raw = american_to_prob(df["home_ml_close"])
    total = p_vis_raw + p_home_raw
    df["p_market_home"] = p_home_raw / total
    df["vig"] = total - 1.0

    print(f"\ntotal games: {len(df)}", flush=True)
    print(f"duplicate game_ids: {df['game_id'].duplicated().sum()}", flush=True)
    print(f"missing closing line: {df['vis_ml_close'].isna().sum()}", flush=True)
    print(f"\nmedian vig: {df['vig'].median():.4f}", flush=True)
    print(f"market home win prob (mean): {df['p_market_home'].mean():.4f}", flush=True)
    print(f"actual home win rate:        "
          f"{(df['home_score_src'] > df['vis_score_src']).mean():.4f}", flush=True)

    print("\ndoubleheaders per season:", flush=True)
    print(df[df["game_num"] > 0].groupby("season").size().to_string(), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()