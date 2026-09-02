# MLB Game Forecasting

Forecasting MLB game outcomes with strictly point-in-time features, evaluated
walk-forward against calibrated baselines.

The goal here is not a high accuracy number. It is a pipeline where every
prediction is made using only information available before first pitch, and
where every result is measured against a baseline that is hard to beat.

## Headline result

Evaluated on 17,008 games (2017-2024), predicting the home team's win probability:

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Base rate (league home win %) | 0.69131 | 0.24908 | 53.0% |
| Elo | 0.67718 | 0.24212 | 57.3% |
| Logistic regression (Elo + pitcher/team features) | 0.67661 | 0.24184 | 57.3% |

**The added features barely improved on Elo.** The log loss gain is 0.0006 and
one season out of seven was negative — that is noise, not signal. This is
reported as the result rather than buried, because it is the result.

The most likely explanation is redundancy: a team's Elo rating is largely driven
by its recent run differential, so a rolling run-differential feature carries
little new information. The fitted coefficients agree — `elo_diff` sits around
0.28 while `team_net_diff` is 0.05.

## Design: how leakage is prevented

Sports models are easy to accidentally inflate. Season-long statistics computed
over a full season and then used to predict games *within* that season will
produce impressive-looking accuracy that does not exist out of sample. Three
choices rule this out structurally rather than by inspection:

1. **Single chronological pass.** Features are built in one loop ordered by
   date. Each row's features are emitted from accumulator state, and only then
   is that game's result folded into the state. A game can never inform its own
   prediction.
2. **Walk-forward evaluation.** The model trains on seasons strictly before the
   season it predicts, refit each year. No random train/test split is used
   anywhere.
3. **No post-game columns.** The raw source includes fields like team rank,
   games back, and streak, all of which already contain the outcome of the row
   they sit on. None are used.

## Data

Game logs from [Retrosheet](https://www.retrosheet.org/gamelogs/), 2015-2024,
one row per game with both starting pitchers identified.

**2020 is excluded.** The 60-game season had a universal DH, seven-inning
doubleheaders, and a runner placed on second base in extra innings. It is a
different game, and including it would corrupt any rolling feature spanning it.

21,865 games after exclusion, with zero duplicate game IDs (doubleheaders are
disambiguated by date, teams, and game number).

## Findings

**Home-field advantage is declining.** Home win rate falls from .5414 in 2015 to
.5216 in 2024. A fixed home-field constant would be miscalibrated across this
span, which is why the Elo home edge is expressed in rating points rather than
as a fixed probability bump.

**Days of rest has a small negative association with winning** (r = -0.023 for
the home-minus-visitor rest differential). This is opposite the intuitive
direction and is unlikely to be causal. Extra rest is confounded with the
reasons it occurs: a skipped rotation slot, a starter pushed back for a minor
issue, or a team returning from travel. Raw rest values also required capping —
offseason and injury gaps reached 2,900 days and would otherwise read to a model
as extreme freshness.

**Career start counts were deliberately excluded.** The count increases
monotonically with the calendar, so a model can use it as a proxy for date. It
is retained only as a saturating flag for whether a pitcher has enough history
for the rolling average to be meaningful (89% of starts qualify).

**Elo is well calibrated.** In the buckets carrying real sample size
(0.4-0.7, ~15,600 games), predicted and actual win rates differ by under 1.5
percentage points. The extreme buckets contain a handful of games each and their
apparent gaps are noise.

## Known limitations

- **Starting pitcher quality is a proxy.** Retrosheet game logs carry team-level
  run totals, so "runs allowed in this pitcher's starts" includes bullpen
  innings. True per-start lines require the event files or Statcast.
- **No betting market comparison yet.** The strongest available benchmark is the
  closing moneyline, since it represents a large, liquid, well-informed
  consensus. Beating the base rate is easy; beating the closing line is the real
  test, and it has not been attempted here.
- **No bullpen, park, or weather features.** Run environment varies substantially
  by park, and reliever fatigue is unmodeled.

## Next steps

1. Join historical closing moneylines and evaluate against the market, including
   closing line value.
2. Replace the pitcher proxy with true per-start lines from event files.
3. Add bullpen usage over the prior three days and park run environment.
4. Gradient boosting, once there are enough non-redundant features for it to
   have something to work with.

## Repo layout

| File | Purpose |
|---|---|
| `src/fetch_gamelogs.py` | Download one season of Retrosheet game logs to parquet |
| `src/build_spine.py` | Combine seasons, build unique game IDs, sort by date |
| `src/build_features.py` | Single chronological pass producing point-in-time features |
| `src/metrics.py` | Log loss, Brier, accuracy, calibration table |
| `src/baseline_elo.py` | Walk-forward Elo baseline |
| `src/train_model.py` | Walk-forward logistic regression vs. Elo |

## Reproducing

Create the environment:

    conda create -n mlb python=3.12 -y
    conda activate mlb
    pip install pybaseball duckdb pandas pyarrow scikit-learn matplotlib

Download the seasons (bash):

    for y in 2015 2016 2017 2018 2019 2021 2022 2023 2024; do
      python src/fetch_gamelogs.py $y
    done

Or on Windows:

    for %y in (2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_gamelogs.py %y

Then build and evaluate:

    python src/build_spine.py
    python src/build_features.py
    python src/baseline_elo.py
    python src/train_model.py

Data files are not committed; the scripts above download and rebuild everything.

## Attribution

The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark,
DE 19711.
