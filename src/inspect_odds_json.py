"""Inspect the JSON odds dataset before writing a parser against it."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "raw" / "odds" / "mlb_odds_dataset.json"


def main():
    print(f"loading {PATH.name} ({PATH.stat().st_size / 1e6:.0f} MB)...", flush=True)
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dates = sorted(data.keys())
    print(f"dates: {len(dates):,}  from {dates[0]} to {dates[-1]}\n", flush=True)

    games_per_year = Counter(d[:4] for d in dates for _ in data[d])
    print("games per year (unfiltered):", flush=True)
    for y in sorted(games_per_year):
        print(f"  {y}: {games_per_year[y]:,}")

    gametypes = Counter()
    books = Counter()
    n_books = Counter()
    for d in dates:
        for g in data[d]:
            gametypes[g.get("gameView", {}).get("gameType")] += 1
            ml = g.get("odds", {}).get("moneyline", []) or []
            n_books[len(ml)] += 1
            for entry in ml:
                books[entry.get("sportsbook")] += 1

    print(f"\ngame types: {dict(gametypes)}", flush=True)
    print(f"\nsportsbooks seen:", flush=True)
    for b, c in books.most_common():
        print(f"  {b}: {c:,}")
    print(f"\nmoneyline entries per game: "
          f"{dict(sorted(n_books.items()))}", flush=True)

    # One full example
    sample_date = "2023-06-15" if "2023-06-15" in data else dates[len(dates) // 2]
    print(f"\nfull example game from {sample_date}:", flush=True)
    print(json.dumps(data[sample_date][0], indent=2)[:2500], flush=True)


if __name__ == "__main__":
    main()