"""Pull one team-season schedule from Baseball Reference to inspect the columns."""
from pybaseball import schedule_and_record

def main():
    print("fetching...", flush=True)
    df = schedule_and_record(2024, "CHC")
    print(f"shape: {df.shape}", flush=True)
    print("\ncolumns:", flush=True)
    for c in df.columns:
        print(f"  {c}")
    print("\nfirst 3 rows:", flush=True)
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()