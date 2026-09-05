# MLB Game Forecasting

Forecasting MLB game outcomes with strictly point-in-time features, evaluated
walk-forward against the closing betting line across fifteen seasons.

The goal is not a high accuracy number. It is a pipeline where every prediction
uses only information available before first pitch, every metric is included
only after its reliability has been measured, every hypothesis is registered
before it is tested, and the benchmark is one that is genuinely hard to beat.

## Headline result

Evaluated on 27,992 games (2012-2024) with closing moneylines attached,
predicting the home team's win probability:

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Base rate (league home win %) | 0.69098 | 0.24892 | 53.3% |
| Elo | 0.68036 | 0.24367 | 56.5% |
| Team-level features (runs-allowed proxy) | 0.67995 | 0.24347 | 56.4% |
| Statcast pitcher features | 0.67869 | 0.24286 | 56.8% |
| **Market (closing moneyline)** | **0.67318** | **0.24020** | **58.1%** |
| Statcast features + market price | 0.67346 | 0.24033 | 58.0% |

**The market wins.** Adding every engineered feature on top of the market price
makes the prediction slightly *worse*. The model's own features carry real
signal — they beat the base rate decisively — but that information is already
inside the closing line.

Seven pre-registered subsets were tested for local mispricing. None beat the
market. One hypothesis was tested three times as the dataset grew and was
rejected when its effect decayed toward zero.

## What this project is actually demonstrating

Most sports models report accuracy against a weak baseline. This one is built
around four practices that are harder to fake:

1. **Metrics are screened for reliability before use**, by measured split-half
   correlation, not by plausibility or in-sample fit.
2. **Hypotheses are pre-registered with stated mechanisms**, and one season is
   sealed in a lockbox. The lockbox was never opened, because it was never
   needed.
3. **Every join is validated against an independent source.** This caught three
   separate bugs that no amount of inspecting a single source would have found.
4. **Negative results are the headline.** The market wins, the promising
   hypothesis died, and both are reported as the finding.

## Which pitcher metrics are worth using

Before building any pitcher feature, each candidate was tested for split-half
reliability: for a pitcher with at least 2k starts in a season, split k starts
into each of two piles at random and correlate the metric across piles. High
correlation means the metric measures the pitcher; low correlation means it
measures luck.

Measured at k = 8, the rolling window the model uses:

| Metric | r at k=8 | Verdict |
|---|---|---|
| Release velocity | **0.978** | Physical measurement; near-perfect |
| Strikeout rate | **0.645** | Stabilized |
| Walk rate | 0.427 | Nearly stabilized |
| xwOBA on contact | 0.392 | Marginal |
| Run expectancy per batter faced | 0.230 | Mostly noise at this window |
| Home run rate | 0.160 | Noise |

**Descriptive accuracy and predictive value run opposite here.** Run expectancy
per batter faced is the best available description of what a pitcher did.
Sorting all 68,020 starts by it returns, in order: Verlander's 2019 no-hitter,
German's 2023 perfect game, Braden's 2010 perfect game, Halladay's 2010 perfect
game, Cain's 2012 perfect game, and Hernandez's 2012 perfect game. The metric
independently rediscovers the best pitching performances of the last fifteen
years without being told what a no-hitter is.

It is also the *least* reliable metric in the table at an 8-start window,
because it embeds batted-ball outcomes that depend on defense, park, and luck.
It was excluded from the model, along with home run rate — whose 0.160
reliability independently replicates the well-known result that pitchers exert
little control over home runs per fly ball.

xwOBA was also dropped, for a different reason: it requires Statcast tracking
and does not exist before 2015. Removing it cost 0.00003 log loss and gained
five seasons of clean modeling.

Velocity is carried as a **deviation from each pitcher's own baseline** rather
than as a level. The level is reliable but says little about winning; a drop
against a pitcher's own norm is a low-noise signal of fatigue or injury.

## Testing the margins

Overall market efficiency does not rule out local inefficiency. But subset
search is where honest projects become dishonest ones — test twenty slices at
the 5% level and one will look significant from noise alone.

Two safeguards: every subset was specified in code with a stated mechanism
*before* results were examined, and the 2024 season was sealed and never
examined.

Exploration set 2012-2023, 25,752 games. "Gain" is market log loss minus model
log loss, so positive means the model beat the market. CIs from 2,000 bootstrap
resamples:

| Subset | Mechanism | n | Gain | 95% CI | Beats market |
|---|---|---|---|---|---|
| All games | reference | 25,752 | -0.0055 | (-0.0067, -0.0043) | no |
| First 15 games of season | little current-season info | 4,935 | -0.0042 | (-0.0069, -0.0012) | no |
| Unproven starter | no track record to price | 3,566 | -0.0088 | (-0.0125, -0.0049) | no |
| Doubleheader game 2 | bullpen depletion unpriced | 209 | -0.0109 | (-0.0289, +0.0070) | no |
| Short-rest starter | non-standard rotation slot | 231 | -0.0027 | (-0.0210, +0.0166) | no |
| Widest vig quintile | thin, low-confidence market | 5,151 | -0.0042 | (-0.0070, -0.0015) | no |
| Largest disagreement decile | where our info differs most | 2,576 | -0.0245 | (-0.0328, -0.0161) | no |

### The disagreement decile is the central finding

The subset analysis was run three times: with weak team-level features, after
the Statcast upgrade, and again on the expanded fifteen-season dataset. Every
subset improved with better features — except one.

| Subset | Weak features | Statcast features |
|---|---|---|
| All games | -0.00547 | -0.00436 |
| First 15 games | -0.00352 | -0.00197 |
| Unproven starter | -0.00700 | -0.00620 |
| Widest vig quintile | -0.00941 | -0.00344 |
| **Largest disagreement decile** | **-0.01824** | **-0.01979** |

Where the model diverges most from the closing line, a *better* model performs
*worse*. On the full dataset this is the tightest and most decisive negative in
the project: -0.0245 with a confidence interval nowhere near zero.

If the features contained information the market lacked, the disagreement
decile is precisely where it would surface. That improving the model sharpens
the deficit rather than closing it is direct evidence that the disagreement is
noise — and that the market's advantage does not come from a feature that could
simply be added.

### A hypothesis that died correctly

**Mechanism, stated before testing:** velocity is publicly observable but is not
a headline number, and books price primarily off results. A starter throwing 2+
mph below his own baseline shows a physical signal of fatigue or injury before
it appears in his ERA. If the market is slow to incorporate it, that team should
win *less* often than the closing line implies.

Velocity delta is the most reliable measurement available (split-half r =
0.978), so a 2 mph deviation is signal rather than noise.

The test was run three times as odds coverage expanded:

| Dataset | Team-games | Bias at -2.0 mph | Bias at -3.0 mph |
|---|---|---|---|
| 6 seasons | 21,367 | -0.0202 | untestable (n=40) |
| 8 seasons | 32,913 | -0.0114 | -0.0525 (n=178) |
| **13 seasons** | **55,391** | **-0.0036** | **-0.0234 (n=647)** |

**The effect shrank by 82% as the sample tripled.** A real effect holds its
magnitude and tightens its interval as data accumulates. This one decayed toward
zero — the signature of regression to the mean.

The hypothesis is rejected. The lockbox was never opened, because it was never
needed: more data answered the question more convincingly than a single
confirmation test could have.

The early result was directionally correct, monotonic across thresholds, and
entirely spurious. Stopping at six seasons would have produced a "finding."

### Why the deciles are in the output

The velocity analysis also reports deciles of velocity delta. In the six-season
run, decile 4 — a trivial -0.09 to -0.26 mph range with no plausible mechanism —
showed a bias of -0.019, the same magnitude as the pre-registered primary test,
with a confidence interval that nearly excluded zero. On the full dataset it is
+0.0016.

That is what noise looks like. Had the analysis gone fishing across deciles
instead of committing to a threshold in advance, decile 4 would have been
reported as a finding. It is left in the output deliberately.

## Design: how leakage is prevented

Season-long statistics computed over a full season and then used to predict
games *within* that season produce impressive accuracy that does not exist out
of sample. Three choices rule this out structurally:

1. **Single chronological pass.** Features are built in one date-ordered loop.
   Each row's features are emitted from accumulator state, and only then is that
   game's result folded into the state. A game cannot inform its own prediction.
2. **Walk-forward evaluation.** Models train on seasons strictly before the
   season they predict, refit each year. No random train/test split is used
   anywhere.
3. **No post-game columns.** The raw source includes team rank, games back, and
   streak — all of which already contain the outcome of the row they sit on.

Rate metrics are accumulated as summed numerators over summed denominators, and
every metric tracks its numerator and denominator as a **matched pair** — a
start missing a metric contributes to neither side. An earlier version counted a
missing xwOBA as zero while still counting its batters, which silently recorded
every pre-2015 start as a perfect performance.

**Career start counts were deliberately excluded.** The count rises
monotonically with the calendar, making it a proxy for date that a model will
happily exploit. It survives only as a saturating flag for whether a pitcher has
enough history for a rolling average to mean anything.

## Three bugs caught by independent validation

Every join in this pipeline is checked against a source that was produced
independently. Three real bugs surfaced this way, none of which would have been
visible from inspecting a single source.

**Doubleheaders swapped.** The initial odds join matched 99.86% of games but
agreed on final score for only 98.75%. The failures were almost all
doubleheaders with game 1 and game 2 reversed: the odds file orders by betting
rotation number, which does not follow Retrosheet's scheduled sequence. The fix
emits both orderings and keeps whichever the independently-sourced score
confirms. Agreement rose to 99.83%.

**Spring training contamination.** The Statcast pull initially returned 2,554
games for a 2,429-game season, and 1,067 pitchers where only 831 appeared in the
regular season. Spring training stats would have corrupted every rolling
pitcher average.

**A median of American odds is meaningless.** American odds are discontinuous —
no valid line exists strictly between -100 and +100. Taking a median across
books straddling that gap produces arithmetic nonsense: four books at
[-110, -104, +100, +105] give a "median" of -2, which is not a price. This
surfaced only because the JSON dataset could be compared against the Excel
archive on their 2021 overlap: correlation was 0.628 when it should have been
near 1. Computing consensus in probability space instead raised it to **0.9975**,
with 98.6% of games agreeing within 2 probability points.

## Data

**Game logs:** [Retrosheet](https://www.retrosheet.org/gamelogs/), 2010-2024 —
one row per game with both starting pitchers. 34,015 games, zero duplicate IDs.

**Statcast / PITCHf/x:** 9.99 million pitches via pybaseball, regular season
only, aggregated to 68,020 starting-pitcher outings. Release velocity and run
expectancy are available from 2010; xwOBA, launch speed and spin rate require
Statcast cameras and begin in 2015.

**Odds:** two independent sources, unioned.

- 2010-2019: SportsBookReview Excel archive, one aggregated closing line.
- 2021-2024: JSON dataset (median of ~6 books, de-vigged per book in
  probability space). Games where books disagreed by more than 10 probability
  points were dropped as unreliable.

32,523 of 34,015 games matched to a closing price (95.6%), with 99.83% score
agreement against Retrosheet.

**2020 is excluded.** The 60-game season had a universal DH, seven-inning
doubleheaders, and a runner placed on second in extra innings.

Odds files must be downloaded manually — both sources block automated access.
See `data/README.md`.

## Findings about the game and its market

**Home-field advantage has declined sharply**, from .5593 in 2010 to .5216 in
2024. A fixed home-field constant would be badly miscalibrated across this span.

**Bookmaker hold has risen with the retail era.** Median vig is 2.79% across the
2010-2019 aggregated lines and 4.22% across the 2021-2024 retail books. This
does not affect de-vigged probability, but it is a real change in market
structure over the window.

**The market is extremely well calibrated.** Across every probability bucket
with meaningful sample, predicted and actual win rates differ by under 1.1
percentage points; in the largest bucket (10,855 games) the gap is 0.0000.

**Short-rest games are hard for everyone.** Market log loss on starters with 4
or fewer days rest is 0.698 against 0.673 overall — the market's own accuracy
degrades on non-standard rotation slots, though not enough to create an edge.

**Days of rest has a small negative association with winning** (r = -0.016),
opposite the intuitive direction and unlikely to be causal. Rest is confounded
with the reasons it occurs: a skipped rotation slot, a starter pushed back for a
minor issue, a team returning from travel.

**One pre-registered mechanism was falsified.** Wide vig was predicted to signal
thin, low-confidence markets. Those games are in fact easier to predict: wide
vig tracks lopsided matchups, where books charge more on heavy favorites.

## Known limitations

- **The JSON odds sample is not random.** Games dropped for book disagreement
  are systematically those books found hardest to price, which is plausibly
  where mispricing lives. An unreliable price is worse than no price, but the
  remaining sample should not be treated as representative.
- **Innings pitched is approximated.** Statcast does not expose outs recorded
  per pitcher, so workload is inferred from batters faced and last inning.
- **Velocity baselines drift with age.** The baseline is a career average
  dominated by peak years, so recent velocity sits systematically below it
  (median -0.13 mph). Within-pitcher comparisons are unaffected, but a rolling
  prior-season baseline would be cleaner.
- **2010 and 2011 match at only 93%** against the odds archive, versus 99%+ for
  2012-2019. Not diagnosed.
- **No bullpen, park, or weather features.**

## Next steps

1. Bullpen usage over the prior three days, and park run environment — both
   screened for reliability before inclusion.
2. Gradient boosting and stacking. Expected to be marginal on seven mostly
   linear features, but untested.
3. Prior-season velocity baselines instead of career.

## Repo layout

| File | Purpose |
|---|---|
| `src/fetch_gamelogs.py` | Download Retrosheet game logs to parquet |
| `src/build_spine.py` | Combine seasons, build unique game IDs, sort by date |
| `src/build_features.py` | Point-in-time team and Elo features |
| `src/inspect_odds.py` | Inspect an odds workbook before parsing |
| `src/build_odds.py` | Parse Excel archive, map team codes, de-vig |
| `src/inspect_odds_json.py` | Inspect the JSON odds dataset |
| `src/build_odds_json.py` | Parse JSON odds; consensus in probability space |
| `src/compare_odds_sources.py` | Cross-source price validation on 2021 |
| `src/join_odds.py` | Union both sources; validate against scores |
| `src/fetch_statcast.py` | Pull pitch data, cached and resumable |
| `src/build_pitcher_starts.py` | Aggregate pitches into per-start lines |
| `src/stabilization.py` | Split-half reliability by sample size |
| `src/build_pitcher_features.py` | Point-in-time pitcher features |
| `src/metrics.py` | Log loss, Brier, accuracy, calibration |
| `src/baseline_elo.py` | Walk-forward Elo baseline |
| `src/train_model.py` | Walk-forward logistic regression |
| `src/evaluate_market.py` | Model and baselines vs. the closing line |
| `src/evaluate_pitcher.py` | Statcast features vs. the old proxy |
| `src/find_edges.py` | Pre-registered subset analysis |
| `src/velocity_edge.py` | Pre-registered velocity-decline hypothesis test |

## Reproducing

    conda create -n mlb python=3.12 -y
    conda activate mlb
    pip install pybaseball pandas pyarrow scikit-learn matplotlib openpyxl

Download game logs and pitch data (Windows):

    for %y in (2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_gamelogs.py %y
    for %y in (2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_statcast.py %y

Download odds manually into `data/raw/odds/` (see `data/README.md`), then:

    python src/build_spine.py
    python src/build_features.py
    python src/build_odds.py
    python src/build_odds_json.py
    python src/compare_odds_sources.py
    python src/join_odds.py
    python src/build_pitcher_starts.py
    python src/stabilization.py
    python src/build_pitcher_features.py
    python src/evaluate_market.py
    python src/evaluate_pitcher.py
    python src/find_edges.py
    python src/velocity_edge.py

Data files are not committed. Retrosheet and Statcast downloads are scripted;
odds workbooks are not.

## Attribution

The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark,
DE 19711.

Statcast data is provided by MLB Advanced Media via Baseball Savant, accessed
through [pybaseball](https://github.com/jldbc/pybaseball).