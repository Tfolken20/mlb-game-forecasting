# MLB Game Forecasting

Forecasting MLB game outcomes with strictly point-in-time features, evaluated
walk-forward against the closing betting line across fifteen seasons.

The goal is not a high accuracy number. It is a pipeline where every prediction
uses only information available before first pitch, every metric is screened for
reliability before it is used, every hypothesis is registered before it is
tested, and the benchmark is one that is genuinely hard to beat.

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
makes the prediction slightly *worse*. The model's features carry real signal —
they beat the base rate decisively — but that information is already inside the
closing line.

## Four independent attempts to find an edge, and four failures

The interesting part of this project is not the headline number. It is that the
same conclusion was reached four separate ways, each attacking a different
possible explanation for the gap.

| Attempt | Premise | Result |
|---|---|---|
| **Better features** | The market knows something we don't measure | Statcast pitcher metrics closed a quarter of the gap, then stopped |
| **A reliable signal the market ignores** | Velocity decline is observable but not headline news | Effect shrank 82% as data tripled; rejected |
| **A mechanism nobody prices** | Bullpen depletion is invisible and unposted | 98% redundant with team strength; rejected |
| **A better model class** | The relationships are nonlinear | Gradient boosting was *worse*; stacking was a tie |
| **A second-moment effect** | Park changes variance, not strength | Market prices it fully; rejected |

Each is documented below with its pre-registered mechanism and its data.

## Which pitcher metrics are worth using

Before building any pitcher feature, each candidate was tested for split-half
reliability: for a pitcher with at least 2k starts in a season, split k starts
into each of two piles at random and correlate the metric across piles.

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
game, Cain's 2012 perfect game, and Hernandez's 2012 perfect game — every
perfect game of the era, surfaced by a metric never told what one is.

It is also the *least* reliable metric in the table at an 8-start window,
because it embeds batted-ball outcomes that depend on defense, park, and luck.
It was excluded, along with home run rate — whose 0.160 reliability
independently replicates the known result that pitchers exert little control
over home runs per fly ball.

## Screening kills features that look promising

Reliability screening is not a formality. Two feature families passed a
plausibility check and failed the data.

### Bullpen workload: a mechanism that turned out to be a proxy

**Mechanism:** a bullpen that threw heavily over the previous two or three days
has fewer rested high-leverage arms. Books price starters explicitly and
publicly; relief availability is posted nowhere.

Relief pitches were computed per team per game from 10 million Statcast pitches
and rolled forward by *calendar day*, since an off day restores a bullpen and a
doubleheader depletes it twice. Four checks:

| Check | Result | Reading |
|---|---|---|
| Does depletion persist to the next game? | r = -0.019 to +0.022 | No |
| Does it predict a longer leash on the starter? | r = **-0.055** | Wrong sign |
| Raw association with winning | r = +0.027 | Looks promising |
| **Residual after removing Elo and run differential** | **r = +0.0005** | **Entirely redundant** |

The raw correlation of +0.027 would have looked like a modest real signal to
anyone who checked only that. But bullpen workload correlates **+0.22 with team
run differential** and **+0.16 with Elo** — bad teams burn their bullpens. It is
a team-quality proxy wearing a fatigue label, and 98% of its apparent signal
disappears once team strength is removed.

The feature was not added. Unscreened, it would have added variance under a
misleading name and been nearly impossible to diagnose later.

### Park factors: the market already prices the variance effect

**Mechanism:** park does not favor the home team — both sides bat in it. What a
hitter's park changes is *variance*. A higher run environment widens the
distribution of run differential, compressing win probability toward .500. A
team that should win 60% in a neutral park should win somewhat less in Coors.
If the market under-adjusts, its probabilities will be too extreme in high-run
parks.

This is an interaction hypothesis and was tested as one, walk-forward:

    logit(P(home win)) = a + b·market_logit + c·market_logit·park_centered

The mechanism predicts c < 0. Result: **c = -0.0025, 95% CI (-1.01, +0.78).**
Null. The quintile table is flat, with biases of -0.002, +0.004, +0.010, -0.000
and +0.003 against a predicted monotone decline. Coors Field alone, the most
extreme park in baseball, shows favorites at -0.008 against implied on 590
games.

Point-in-time park factors were computed on an expanding window with shrinkage
toward league average, and validated by recovering baseball geography
independently: the five lowest are Oracle, T-Mobile, Petco, Tropicana and Citi
Field; the five highest are Chase, Fenway, Globe Life and Coors at 1.24.

**The market fully prices a second-moment effect.** That is a more interesting
null than a first-order one: the closing line is not merely getting team
strength right, it correctly handles how run environment reshapes the outcome
distribution.

One diagnostic worth recording: the 2013 fold produced an interaction
coefficient of -1.89 against -0.0003 to +0.0050 in every later fold. That fold
trains on 2010-2012, when the expanding park calculation has the least history,
and an unpenalized fit latched onto the noise. A useful illustration of why thin
data and unpenalized models are a bad combination.

## Model class is not the constraint

Three families were compared under identical walk-forward evaluation, with
stacking implemented using **nested splits** — the meta-learner trains only on
out-of-fold base predictions, since fitting base and meta on the same rows
teaches the meta to trust an overfit signal.

Evaluated on 25,752 games (2012-2023):

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Logistic regression | **0.67854** | 0.24278 | 56.8% |
| Gradient boosting | 0.68002 | 0.24350 | 56.8% |
| Stacked (logistic + GBM) | 0.67820 | 0.24262 | 57.0% |
| Market | 0.67305 | 0.24013 | 58.1% |

**Gradient boosting is worse than logistic regression** — by 0.0015 log loss,
and worse in 9 of 11 seasons. Its calibration is worse in the buckets that
matter: off by 1.3 points in the 0.4-0.5 range across 6,819 games.

Stacking is the decisive evidence. A meta-learner given both models had every
opportunity to weight the GBM where it helped, and effectively reproduced the
logistic model instead. The GBM contributed no independent information.

With seven features whose relationships to the outcome are essentially linear
and monotone, extra model capacity buys nothing and costs variance.

## Testing the margins

Overall efficiency does not rule out local inefficiency. But subset search is
where honest projects become dishonest ones — test twenty slices at the 5% level
and one will look significant from noise alone.

Two safeguards: every subset was specified in code with a stated mechanism
*before* results were examined, and the 2024 season was sealed. **The lockbox
was never opened, because it was never needed.**

Exploration set 2012-2023, 25,752 games. "Gain" is market log loss minus model
log loss, so positive means the model beat the market:

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
the Statcast upgrade, and on the full fifteen-season dataset. Every subset
improved with better features — except one.

| Subset | Weak features | Statcast features |
|---|---|---|
| All games | -0.00547 | -0.00436 |
| First 15 games | -0.00352 | -0.00197 |
| Unproven starter | -0.00700 | -0.00620 |
| Widest vig quintile | -0.00941 | -0.00344 |
| **Largest disagreement decile** | **-0.01824** | **-0.01979** |

Where the model diverges most from the closing line, a *better* model performs
*worse*. On the full dataset this is the tightest negative in the project:
-0.0245, with a confidence interval nowhere near zero.

If the features contained information the market lacked, the disagreement decile
is precisely where it would surface. That improving the model sharpens the
deficit rather than closing it is direct evidence that the disagreement is noise
— and that the market's advantage is not a feature that could simply be added.

### A hypothesis that died correctly

**Mechanism:** velocity is publicly observable but not a headline number, and
books price primarily off results. A starter throwing 2+ mph below his own
baseline shows a physical signal of fatigue or injury before it reaches his ERA.
Velocity delta is the most reliable measurement available (r = 0.978), so a 2
mph deviation is signal, not noise.

The test was run three times as odds coverage expanded:

| Dataset | Team-games | Bias at -2.0 mph | Bias at -3.0 mph |
|---|---|---|---|
| 6 seasons | 21,367 | -0.0202 | untestable (n=40) |
| 8 seasons | 32,913 | -0.0114 | -0.0525 (n=178) |
| **13 seasons** | **55,391** | **-0.0036** | **-0.0234 (n=647)** |

**The effect shrank by 82% as the sample tripled.** A real effect holds its
magnitude and tightens its interval. This one decayed toward zero — the
signature of regression to the mean.

The early result was directionally correct, monotonic across thresholds, and
entirely spurious. Stopping at six seasons would have produced a "finding."

### Why the deciles are left in the output

The velocity analysis also reports deciles of velocity delta. In the six-season
run, decile 4 — a trivial -0.09 to -0.26 mph range with no plausible mechanism —
showed a bias of -0.019, the same magnitude as the pre-registered primary test,
with an interval that nearly excluded zero. On the full dataset it is +0.0016.

That is what noise looks like. Had the analysis gone fishing across deciles
instead of committing to a threshold in advance, decile 4 would have been
reported as a finding.

## Design: how leakage is prevented

Season-long statistics computed over a full season and then used to predict
games *within* that season produce impressive accuracy that does not exist out
of sample. Three choices rule this out structurally:

1. **Single chronological pass.** Features are built in one date-ordered loop.
   Each row's features are emitted from accumulator state, and only then is that
   game's result folded into the state. A game cannot inform its own prediction.
2. **Walk-forward evaluation.** Models train on seasons strictly before the
   season they predict. No random train/test split is used anywhere. Stacking
   uses nested splits.
3. **No post-game columns.** The raw source includes team rank, games back, and
   streak — all of which already contain the outcome of the row they sit on.

Rate metrics are accumulated as summed numerators over summed denominators, and
every metric tracks its numerator and denominator as a **matched pair** — a
start missing a metric contributes to neither side. An earlier version counted a
missing xwOBA as zero while still counting its batters, silently recording every
pre-2015 start as a perfect performance.

**Career start counts were deliberately excluded.** The count rises
monotonically with the calendar, making it a proxy for date that a model will
happily exploit.

## Three bugs caught by independent validation

Every join is checked against a source produced independently. Three real bugs
surfaced this way, none visible from inspecting a single source.

**Doubleheaders swapped.** The initial odds join matched 99.86% of games but
agreed on final score for only 98.75%. The failures were almost all
doubleheaders with game 1 and game 2 reversed: the odds file orders by betting
rotation number, which does not follow Retrosheet's scheduled sequence. The fix
emits both orderings and keeps whichever the independent score confirms.
Agreement rose to 99.83%.

**Spring training contamination.** The Statcast pull initially returned 2,554
games for a 2,429-game season and 1,067 pitchers where only 831 appeared in the
regular season. Those stats would have corrupted every rolling pitcher average.

**A median of American odds is meaningless.** American odds are discontinuous —
no valid line exists strictly between -100 and +100. Taking a median across
books straddling that gap produces arithmetic nonsense: four books at
[-110, -104, +100, +105] give a "median" of -2, which is not a price. This
surfaced only because the JSON dataset could be compared against the Excel
archive on their 2021 overlap: correlation was 0.628 where it should have been
near 1. Computing consensus in probability space raised it to **0.9975**, with
98.6% of games agreeing within 2 probability points.

## Data

**Game logs:** [Retrosheet](https://www.retrosheet.org/gamelogs/), 2010-2024 —
one row per game with both starting pitchers. 34,015 games, zero duplicate IDs.

**Statcast / PITCHf/x:** 9.99 million pitches via pybaseball, regular season
only, aggregated to 68,020 starting-pitcher outings. Release velocity and run
expectancy are available from 2010; xwOBA, launch speed and spin rate require
Statcast cameras and begin in 2015.

**Odds:** two independent sources, unioned.

- 2010-2019: SportsBookReview Excel archive, one aggregated closing line.
- 2021-2024: JSON dataset, median across ~6 books, each de-vigged individually
  in probability space. Games where books disagreed by more than 10 probability
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
2010-2019 aggregated lines and 4.22% across the 2021-2024 retail books.

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
thin, low-confidence markets. Those games are in fact easier to predict: wide vig
tracks lopsided matchups, where books charge more on heavy favorites.

## Known limitations

- **The JSON odds sample is not random.** Games dropped for book disagreement
  are systematically those books found hardest to price, which is plausibly
  where mispricing lives. An unreliable price is worse than no price, but the
  remaining sample should not be treated as representative.
- **Innings pitched is approximated.** Statcast does not expose outs recorded
  per pitcher, so workload is inferred from batters faced and last inning.
- **Velocity baselines drift with age.** The baseline is a career average
  dominated by peak years, so recent velocity sits systematically below it
  (median -0.13 mph). A rolling prior-season baseline would be cleaner.
- **2010 and 2011 match at only 93%** against the odds archive, versus 99%+ for
  2012-2019. Not diagnosed.
- **No weather, lineup, or injury features.** These are the most likely sources
  of the market's remaining advantage, and the hardest to obtain historically.

## Next steps

The four failed attempts above point in a consistent direction: the remaining
gap is unlikely to close through better modeling of publicly available box-score
data. What the market has that this model does not is almost certainly
*announced lineups, late scratches, weather at first pitch, and money from
informed participants* — none of which are available in free historical form.

Stated plainly: the productive next step is better data, not a better model.

## Repo layout

| File | Purpose |
|---|---|
| `src/fetch_gamelogs.py` | Download Retrosheet game logs |
| `src/build_spine.py` | Combine seasons, build unique game IDs |
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
| `src/build_bullpen.py` | Bullpen workload by calendar day |
| `src/bullpen_screen.py` | Persistence and redundancy screen (rejected) |
| `src/build_park.py` | Point-in-time park factors with shrinkage |
| `src/park_edge.py` | Variance-compression hypothesis test |
| `src/metrics.py` | Log loss, Brier, accuracy, calibration |
| `src/baseline_elo.py` | Walk-forward Elo baseline |
| `src/train_model.py` | Walk-forward logistic regression |
| `src/evaluate_market.py` | Model and baselines vs. the closing line |
| `src/evaluate_pitcher.py` | Statcast features vs. the old proxy |
| `src/compare_models.py` | Logistic vs. GBM vs. nested stacking |
| `src/find_edges.py` | Pre-registered subset analysis |
| `src/velocity_edge.py` | Pre-registered velocity-decline test |

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
    python src/build_bullpen.py
    python src/bullpen_screen.py
    python src/build_park.py
    python src/evaluate_market.py
    python src/evaluate_pitcher.py
    python src/compare_models.py
    python src/find_edges.py
    python src/velocity_edge.py
    python src/park_edge.py

Data files are not committed. Retrosheet and Statcast downloads are scripted;
odds workbooks are not.

## Attribution

The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark,
DE 19711.

Statcast data is provided by MLB Advanced Media via Baseball Savant, accessed
through [pybaseball](https://github.com/jldbc/pybaseball).