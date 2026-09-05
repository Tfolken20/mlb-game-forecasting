# MLB Game Forecasting

Forecasting MLB game outcomes with strictly point-in-time features, evaluated
walk-forward against the closing betting line.

The goal is not a high accuracy number. It is a pipeline where every prediction
uses only information available before first pitch, every metric is included
only after its reliability has been measured, and every result is scored
against a benchmark that is genuinely hard to beat.

## Headline result

Evaluated on 9,705 games (2017-2021) with closing moneylines attached,
predicting the home team's win probability:

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Elo | 0.67590 | 0.24149 | 57.6% |
| Team-level features (runs-allowed proxy) | 0.67567 | 0.24138 | 57.6% |
| Statcast pitcher features | 0.67413 | 0.24064 | 57.8% |
| **Market (closing moneyline)** | **0.66979** | **0.23857** | **58.7%** |

Replacing a crude team-level pitcher proxy with reliability-screened Statcast
metrics improved the model by 0.0015 log loss and closed roughly a quarter of
the gap to the market.

**The market still wins.** The remaining gap is 0.0044 log loss and 0.9
percentage points of accuracy. No subset tested beat the closing line.

## What makes this project unusual

Most sports models report accuracy against a weak baseline. This one reports
three things that are harder to fake:

1. **Metrics are screened for reliability before use**, not chosen by
   plausibility or by in-sample fit.
2. **Subset hypotheses are pre-registered with stated mechanisms**, and one
   season is sealed in a lockbox that has never been examined.
3. **A negative result is the headline**, including a bug found only because
   two independent data sources were forced to agree.

## Which pitcher metrics are worth using

Before building any pitcher feature, each candidate metric was tested for
split-half reliability: for a pitcher with at least 2k starts in a season,
split k starts into each of two piles at random and correlate the metric
across piles. High correlation means the metric measures the pitcher; low
correlation means it measures luck.

Measured at k = 8, the rolling window the model actually uses (3,165
pitcher-seasons, 43,723 starts):

| Metric | r at k=8 | Verdict |
|---|---|---|
| Release velocity | **0.978** | Physical measurement; near-perfect |
| Strikeout rate | **0.645** | Stabilized |
| Walk rate | 0.427 | Nearly stabilized |
| xwOBA on contact | 0.392 | Marginal |
| Run expectancy per batter faced | 0.230 | Mostly noise at this window |
| Home run rate | 0.160 | Noise |

**Descriptive accuracy and predictive value run opposite here.** Run
expectancy per batter faced is the best available description of what a
pitcher did — sorting 2019 starts by it returns Verlander's no-hitter first,
and the top ten across all seasons includes Germán's perfect game plus the
Musgrove, Rodón, and Kluber no-hitters. It is also the *least* reliable
metric in the table at an 8-start window, because it embeds batted-ball
outcomes that depend on defense, park, and luck.

Run expectancy and home run rate were therefore excluded from the model. The
home-run finding independently replicates the well-known result that pitchers
exert little control over home runs per fly ball.

Velocity is carried as a **deviation from each pitcher's own baseline** rather
than as a level. The level is reliable but says little about winning; a drop
against a pitcher's own norm is a low-noise signal of fatigue or injury.

## Testing the margins

Overall market efficiency does not rule out local inefficiency. But subset
search is where honest projects become dishonest ones — test twenty slices at
the 5% level and one will look significant from noise alone.

Two safeguards:

1. **Pre-registration.** Every subset was specified in code with a stated
   mechanism for *why* mispricing might occur there, before results were
   examined.
2. **A lockbox.** The 2021 season is withheld entirely and has never been
   examined. It exists to confirm one hypothesis, once.

### Subset results

Exploration set 2017-2019, 7,282 games. "Gain" is market log loss minus model
log loss, so positive means the model beat the market. CIs from 2,000
bootstrap resamples:

| Subset | n | Gain | 95% CI | Beats market |
|---|---|---|---|---|
| All games | 7,282 | -0.00436 | (-0.0067, -0.0020) | no |
| First 15 games of season | 1,350 | -0.00197 | (-0.0077, +0.0035) | no |
| Unproven starter | 1,099 | -0.00620 | (-0.0139, +0.0015) | no |
| Widest vig quintile | 1,464 | -0.00344 | (-0.0098, +0.0032) | no |
| Largest disagreement decile | 729 | -0.01979 | (-0.0366, -0.0017) | no |
| Doubleheader game 2 | 95 | — | — | underpowered |
| Short-rest starter | 109 | — | — | underpowered |

### The disagreement decile is the real finding

The subset analysis was run twice: once with the weak team-level features, and
again after the Statcast upgrade. Every subset improved — except one.

| Subset | Weak features | Statcast features |
|---|---|---|
| All games | -0.00547 | -0.00436 |
| First 15 games | -0.00352 | -0.00197 |
| Unproven starter | -0.00700 | -0.00620 |
| Widest vig quintile | -0.00941 | -0.00344 |
| **Largest disagreement decile** | **-0.01824** | **-0.01979** |

Where the model diverges most from the closing line, a *better* model performs
*worse*. It is also the only subset besides the reference whose confidence
interval excludes zero.

If the features contained information the market lacked, the disagreement
decile is precisely where it would surface. That improving the model sharpens
the deficit rather than closing it is direct evidence that the disagreement is
noise — and that the market's advantage does not come from a feature that
could simply be added.

### A pre-registered hypothesis: velocity decline

**Mechanism, stated before testing:** velocity is publicly observable but is
not a headline number, and books price primarily off results. A starter
throwing 2+ mph below his own baseline shows a physical signal of fatigue or
injury before it appears in his ERA. If the market is slow to incorporate it,
that team should win *less* often than the closing line implies.

Velocity delta is the most reliable measurement available (split-half r =
0.978), so a 2 mph deviation is signal rather than noise.

Result on 21,367 team-games with an established baseline (2015-2019):

| Threshold | n | Market implied | Actual | Bias | 95% CI |
|---|---|---|---|---|---|
| ≤ -1.0 mph | 3,144 | 0.5069 | 0.5000 | -0.0069 | (-0.024, +0.010) |
| ≤ -1.5 mph | 1,458 | 0.5046 | 0.4993 | -0.0053 | (-0.030, +0.020) |
| **≤ -2.0 mph (primary)** | **595** | **0.4976** | **0.4773** | **-0.0202** | **(-0.060, +0.019)** |
| ≤ -2.5 mph | 173 | 0.4945 | 0.4682 | -0.0263 | (-0.097, +0.047) |

**Not rejected, not confirmed, underpowered.** The primary test points in the
predicted direction — teams starting a diminished pitcher won 47.7% against an
implied 49.8% — and the effect strengthens monotonically as the signal gets
more extreme, which is what a real effect does and noise usually does not. But
every interval contains zero.

Detecting a two-point bias at conventional power needs on the order of 5,000
observations. The 2.5 mph bucket has 173. This hypothesis is limited by data
volume, not by analysis, which is why extending odds coverage matters more than
any modeling change.

The lockbox was **not** opened. Spending it on a directionally-right but null
result would waste the one clean confirmation available.

### Why the deciles are in the output

The velocity analysis also reports deciles of velocity delta. Decile 4 — a
trivial -0.09 to -0.26 mph range with no plausible mechanism — shows a bias of
-0.019, the same magnitude as the pre-registered primary test, with a CI that
nearly excludes zero.

That is what noise looks like. Had the analysis gone fishing across deciles
instead of committing to a threshold in advance, decile 4 would have been
reported as a finding. It is left in the output deliberately as a
demonstration of why pre-registration is not a formality.

## Design: how leakage is prevented

Season-long statistics computed over a full season and then used to predict
games *within* that season produce impressive accuracy that does not exist out
of sample. Three choices rule this out structurally:

1. **Single chronological pass.** Features are built in one date-ordered loop.
   Each row's features are emitted from accumulator state, and only then is
   that game's result folded into the state. A game cannot inform its own
   prediction.
2. **Walk-forward evaluation.** Models train on seasons strictly before the
   season they predict, refit each year. No random train/test split is used
   anywhere.
3. **No post-game columns.** The raw source includes team rank, games back,
   and streak — all of which already contain the outcome of the row they sit
   on. None are used.

Rate metrics are accumulated as summed numerators over summed denominators
rather than as averages of per-start averages, so a 30-batter start weighs
more than a 12-batter one.

**Career start counts were deliberately excluded.** The count rises
monotonically with the calendar, making it a proxy for date that a model will
happily exploit. It survives only as a saturating flag for whether a pitcher
has enough history for a rolling average to mean anything.

## A bug caught by independent validation

Retrosheet and the odds archive are unrelated sources that both record final
scores. Requiring them to agree — rather than assuming the join was correct —
surfaced a real error.

The initial join matched 99.86% of games but agreed on final score for only
98.75%. The failures were almost all doubleheaders with **game 1 and game 2
swapped**: the odds file orders by betting rotation number, which does not
reliably follow Retrosheet's scheduled game sequence.

The fix emits both orderings and keeps whichever the independently-sourced
score confirms. Agreement rose to 99.82%.

This would not have been visible by inspecting the join. It would have
silently corrupted every market comparison downstream, on exactly the subset
(doubleheaders) most likely to be mispriced.

The 26 residual disagreements are a source quirk on walk-off wins, where the
odds archive sometimes records the score before the final half-inning
completed; the winning team is identical in both sources, so the outcome label
is unaffected. The 21 unmatched games are suspended contests completed on a
later date, which the two sources file differently.

## Data

**Game logs:** [Retrosheet](https://www.retrosheet.org/gamelogs/), 2015-2024 —
one row per game with both starting pitchers identified. 21,865 games, zero
duplicate game IDs.

**Statcast:** pitch-level data via pybaseball, 2015-2024 — 6.46 million
pitches, filtered to regular season only. Spring training was excluded after it
was found to add 125 games and 236 pitchers to a single season; those stats
would corrupt any rolling average. Aggregated to 43,723 starting-pitcher
outings.

**Odds:** SportsBookReview archives, 2015-2021 — opening and closing
moneylines. Free archives stop at 2021. 14,555 of 14,576 games matched
(99.86%).

**2020 is excluded.** The 60-game season had a universal DH, seven-inning
doubleheaders, and a runner placed on second in extra innings. It is a
different game and would corrupt any rolling feature spanning it.

Odds files must be downloaded manually — the source blocks automated access.
See `data/README.md`.

## Other findings

**Home-field advantage is declining**, from .5414 in 2015 to .5216 in 2024. A
fixed home-field constant would be miscalibrated across this span, which is why
the Elo home edge is expressed in rating points rather than a probability bump.

**The market is extremely well calibrated.** Across every probability bucket
with meaningful sample, predicted and actual win rates differ by under 1.2
percentage points.

**Days of rest has a small negative association with winning** (r = -0.023) —
opposite the intuitive direction and unlikely to be causal. Rest is confounded
with the reasons it occurs: a skipped rotation slot, a starter pushed back for
a minor issue, a team returning from travel. Raw values needed capping;
offseason and injury gaps reached 2,900 days and would otherwise read as
extreme freshness.

**One pre-registered mechanism was falsified.** Wide vig was predicted to
signal thin, low-confidence markets. Those games are in fact *easier* to
predict (market log loss 0.640 vs 0.670): wide vig tracks lopsided matchups,
where books charge more on heavy favorites. Reported as falsified rather than
dropped.

## Known limitations

- **Innings pitched is approximated.** Statcast does not expose outs recorded
  per pitcher, so workload is inferred from batters faced and last inning
  appeared.
- **Velocity baselines drift with age.** The baseline is a career average
  dominated by peak years, so recent velocity sits systematically below it
  (median -0.12 mph). This does not affect within-pitcher comparisons but a
  rolling prior-season baseline would be cleaner.
- **No bullpen, park, or weather features.**
- **Odds coverage ends at 2021**, which is what leaves the velocity hypothesis
  and two subset hypotheses underpowered.

## Next steps

1. Extend odds coverage to 2022-2024. This is the binding constraint on every
   open hypothesis, and matters more than any modeling change.
2. Bullpen usage over the prior three days, and park run environment.
3. Prior-season velocity baselines instead of career.
4. Gradient boosting and stacking, once there are enough non-redundant
   features for nonlinearity to have something to work with.

## Repo layout

| File | Purpose |
|---|---|
| `src/fetch_gamelogs.py` | Download Retrosheet game logs to parquet |
| `src/build_spine.py` | Combine seasons, build unique game IDs, sort by date |
| `src/build_features.py` | Point-in-time team and Elo features |
| `src/inspect_odds.py` | Inspect a downloaded odds workbook before parsing |
| `src/build_odds.py` | Parse odds, map team codes, de-vig the closing line |
| `src/join_odds.py` | Join odds and validate against independent scores |
| `src/fetch_statcast.py` | Pull Statcast pitch data, cached and resumable |
| `src/build_pitcher_starts.py` | Aggregate pitches into per-start pitcher lines |
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

Create the environment:

    conda create -n mlb python=3.12 -y
    conda activate mlb
    pip install pybaseball pandas pyarrow scikit-learn matplotlib openpyxl

Download game logs (bash):

    for y in 2015 2016 2017 2018 2019 2021 2022 2023 2024; do
      python src/fetch_gamelogs.py $y
      python src/fetch_statcast.py $y
    done

On Windows:

    for %y in (2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_gamelogs.py %y
    for %y in (2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_statcast.py %y

Download odds workbooks manually into `data/raw/odds/` (see `data/README.md`),
then build and evaluate:

    python src/build_spine.py
    python src/build_features.py
    python src/build_odds.py
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

The information used here was obtained free of charge from and is copyrighted
by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd.,
Newark, DE 19711.

Statcast data is provided by MLB Advanced Media via Baseball Savant, accessed
through [pybaseball](https://github.com/jldbc/pybaseball).