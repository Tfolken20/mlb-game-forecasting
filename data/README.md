# Data

No data files are committed to this repository. This document describes what to
download and where to put it.

## Retrosheet game logs (scripted)

Downloaded automatically. No manual step required.

    for y in 2015 2016 2017 2018 2019 2021 2022 2023 2024; do
      python src/fetch_gamelogs.py $y
    done

Lands in `data/raw/gamelogs_<year>.parquet`.

2020 is deliberately excluded — see the rule changes noted in the project
README.

## Odds archives (manual)

The odds source blocks automated access, so these files must be downloaded by
hand. This is the one step in the pipeline that is not reproducible from code.

**Source:** https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlboddsarchives.htm

**Download:** the season workbooks for 2015, 2016, 2017, 2018, 2019, and 2021.

**Place in:** `data/raw/odds/`

Filenames do not matter as long as they end in `.xlsx` and contain the
four-digit year, which the parser reads from the filename. Files downloaded as
`mlb-odds-2015.xlsx` and similar work as-is.

The free archive ends at 2021. Seasons 2022 onward are not covered, which is
why the market comparison stops there.

## Statcast (scripted, slow)

    python src/fetch_statcast.py 2019

Pulls pitch-level data one month at a time and caches each month in
`data/raw/statcast/`. A full season is roughly 700,000 rows and takes 10-20
minutes. If a month fails, rerun the same command — completed months are read
from cache and skipped.

## Generated files

Everything in `data/processed/` is built by the scripts and can be deleted and
regenerated at any time:

| File | Built by |
|---|---|
| `spine.parquet` | `src/build_spine.py` |
| `features.parquet` | `src/build_features.py` |
| `odds.parquet` | `src/build_odds.py` |
| `modeling.parquet` | `src/join_odds.py` |
| `predictions_elo.parquet` | `src/baseline_elo.py` |
| `predictions_model.parquet` | `src/train_model.py` |
