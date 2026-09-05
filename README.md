# MLB Game Forecasting

Forecasting MLB game outcomes with strictly point-in-time features, evaluated
walk-forward against the closing betting line.

The goal is not a high accuracy number. It is a pipeline where every prediction
uses only information available before first pitch, and where every result is
measured against a benchmark that is genuinely hard to beat.

## Headline result

Evaluated on 9,705 games (2017-2021) with closing moneylines attached,
predicting the home team's win probability:

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Base rate (league home win %) | 0.69085 | 0.24885 | 53.4% |
| Elo | 0.67590 | 0.24149 | 57.6% |
| Own features (Elo + pitcher/team) | 0.67567 | 0.24138 | 57.6% |
| **Market (closing moneyline)** | **0.66979** | **0.23857** | **58.7%** |
| Own features + market price | 0.67024 | 0.23878 | 58.5% |

**The market wins, and the model's features add nothing on top of it.** Feeding
the market price into the model alongside every engineered feature makes the
prediction *worse* by 0.0005 log loss, negative in three of four seasons.

This is the correct result for an efficient market, and it is reported as the
headline rather than buried. The engineered features carry real signal — they
beat the base rate decisively and edge out Elo — but every bit of that
information is already inside the closing line.

## Why this framing

The natural benchmark for a sports model is not accuracy. A model can hit 57%
on home-team picks and still be worthless, because the home team wins about 53%
of the time before any modeling at all.

The closing moneyline is the strongest available benchmark: a large, liquid,
publicly posted consensus that incorporates lineups, injuries, weather, and
sharp money. Beating a naive baseline is easy. Beating the close is the actual
test, and this project reports that test rather than avoiding it.

The market's calibration illustrates the point. Across every probability bucket
with meaningful sample size, predicted and actual win rates differ by under 1.2
percentage points:

| Predicted bucket | n | Predicted | Actual | Gap |
|---|---|---|---|---|
| 0.3-0.4 | 938 | 0.3627 | 0.3742 | +0.0115 |
| 0.4-0.5 | 2,583 | 0.4572 | 0.4533 | -0.0039 |
| 0.5-0.6 | 3,319 | 0.5504 | 0.5538 | +0.0034 |
| 0.6-0.7 | 2,352 | 0.6397 | 0.6318 | -0.0079 |
| 0.7-0.8 | 418 | 0.7294 | 0.7273 | -0.0021 |

## Searching the margins: a pre-registered subset analysis

Overall market efficiency does not rule out local inefficiency. But subset
search is also where honest projects become dishonest ones — test twenty slices
at the 5% level and one will look significant from noise alone.

Two safeguards were applied:

1. **Pre-registration.** Every subset was specified in code, with a stated
   mechanism for *why* mispricing might occur there, before any results were
   examined. A subset without a prior reason is data mining, not a hypothesis.
2. **A lockbox.** The 2021 season was withheld entirely from exploration and
   has not been examined. It exists to confirm a single hypothesis, once.

Results on the exploration set (2017-2019, 7,282 games). "Gain" is market log
loss minus model log loss, so positive means the model beat the market.
Confidence intervals are from 2,000 bootstrap resamples:

| Subset | n | Gain | 95% CI | Beats market |
|---|---|---|---|---|
| All games | 7,282 | -0.00547 | (-0.0081, -0.0029) | no |
| First 15 games of season | 1,350 | -0.00352 | (-0.0099, +0.0031) | no |
| Unproven starter (<5 career starts) | 1,099 | -0.00700 | (-0.0137, +0.0003) | no |
| Widest vig quintile | 1,464 | -0.00941 | (-0.0171, -0.0021) | no |
| Largest disagreement decile | 729 | -0.01824 | (-0.0373, +0.0003) | no |
| Doubleheader game 2 | 95 | — | — | underpowered |
| Short-rest starter | 109 | — | — | underpowered |

**No subset beat the market.** Three findings from the attempt:

**The disagreement decile is diagnostic.** Where the model diverges most from
the closing line, it performs *worst* (-0.018, the largest deficit in the
table). If the features contained information the market lacked, this is
precisely where it would surface. That it does the opposite is clean evidence
that the disagreement is noise, and that the binding constraint is feature
quality — not model form, and not subset selection.

**One pre-registered mechanism was simply wrong.** Wide vig was predicted to
signal thin, low-confidence markets. In fact those games are *easier* to
predict (market log loss 0.640 vs 0.670 overall): wide vig tracks lopsided
matchups, where books charge more on heavy favorites, not uncertainty. The
hypothesis is reported as falsified rather than dropped.

**Two hypotheses remain open, not rejected.** Doubleheader second games and
short-rest starters had fewer than 200 games each in the exploration window.
They are untested, and become testable once odds coverage extends.

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
   None are used.

## A bug caught by independent validation

Retrosheet and the odds archive are unrelated sources that both record final
scores. Joining on score agreement rather than assuming the join was correct
surfaced a real error.

The initial join matched 99.86% of games but agreed on final score for only
98.75%. The failures were almost all doubleheaders with **game 1 and game 2
swapped**: the odds file orders by betting rotation number, which does not
reliably follow Retrosheet's scheduled game sequence.

The fix emits both orderings and keeps whichever the independently-sourced score
confirms. Agreement rose to 99.82%.

This error would not have been visible by inspecting the join. It would have
silently corrupted every market comparison downstream, on exactly the subset
(doubleheaders) most likely to be mispriced.

The 26 residual disagreements are a source quirk on walk-off wins, where the
odds archive occasionally records the score before the final half-inning
completed. The winning team is identical in both sources, so the outcome label
is unaffected. The 21 unmatched games are suspended contests completed on a
later date, which the two sources file differently.

## Data

**Game logs:** [Retrosheet](https://www.retrosheet.org/gamelogs/), 2015-2024 —
one row per game with both starting pitchers identified. 21,865 games, zero
duplicate game IDs.

**Odds:** SportsBookReview archives, 2015-2021 — opening and closing moneylines.
Free archives stop at 2021; 2022-2024 remain uncovered. 14,555 of 14,576 games
in covered seasons matched (99.86%).

**2020 is excluded.** The 60-game season had a universal DH, seven-inning
doubleheaders, and a runner placed on second in extra innings. It is a different
game, and would corrupt any rolling feature spanning it.

Odds files must be downloaded manually — the source blocks automated access. See
`data/README.md` for which files and where they go.

## Other findings

**Home-field advantage is declining**, from .5414 in 2015 to .5216 in 2024. A
fixed home-field constant would be miscalibrated across this span, which is why
the Elo home edge is expressed in rating points rather than a fixed probability
bump.

**Days of rest has a small negative association with winning** (r = -0.023 for
the home-minus-visitor differential) — opposite the intuitive direction and
unlikely to be causal. Rest is confounded with the reasons it occurs: a skipped
rotation slot, a starter pushed back for a minor issue, a team returning from
travel. Raw values also required capping; offseason and injury gaps reached
2,900 days and would otherwise read as extreme freshness.

**Career start counts were deliberately excluded.** The count rises
monotonically with the calendar, making it a proxy for date that a model will
happily exploit. It is retained only as a saturating flag for whether a pitcher
has enough history for the rolling average to mean anything (89% of starts
qualify).

## Known limitations

- **Starting pitcher quality is a proxy.** Retrosheet game logs carry
  team-level run totals, so "runs allowed in this pitcher's starts" includes
  bullpen innings. This is the largest known weakness and the current work in
  progress — see next steps.
- **No bullpen, park, or weather features.** Run environment varies
  substantially by park, and reliever fatigue is unmodeled.
- **Odds coverage ends at 2021.** Recent seasons are unpriced, which is also
  what leaves two subset hypotheses underpowered.

## Next steps

1. Replace the pitcher proxy with true per-start lines from Statcast, including
   expected wOBA on contact and run-expectancy change per pitch — metrics that
   strip out defense and sequencing luck.
2. Add bullpen usage over the prior three days and park run environment.
3. Extend odds coverage to 2022-2024 to power the two open subset hypotheses.
4. Gradient boosting and model stacking, once there are enough non-redundant
   features for nonlinearity to have something to work with.

## Repo layout

| File | Purpose |
|---|---|
| `src/fetch_gamelogs.py` | Download one season of Retrosheet game logs to parquet |
| `src/build_spine.py` | Combine seasons, build unique game IDs, sort by date |
| `src/build_features.py` | Single chronological pass producing point-in-time features |
| `src/inspect_odds.py` | Inspect a downloaded odds workbook before parsing it |
| `src/build_odds.py` | Parse odds workbooks, map team codes, de-vig the closing line |
| `src/join_odds.py` | Join odds to features and validate against independent scores |
| `src/metrics.py` | Log loss, Brier, accuracy, calibration table |
| `src/baseline_elo.py` | Walk-forward Elo baseline |
| `src/train_model.py` | Walk-forward logistic regression vs. Elo |
| `src/evaluate_market.py` | Model and baselines vs. the closing line |
| `src/find_edges.py` | Pre-registered subset analysis with bootstrap CIs |
| `src/fetch_statcast.py` | Pull Statcast pitch-level data, cached by month |

## Reproducing

Create the environment:

    conda create -n mlb python=3.12 -y
    conda activate mlb
    pip install pybaseball pandas pyarrow scikit-learn matplotlib openpyxl

Download game logs (bash):

    for y in 2015 2016 2017 2018 2019 2021 2022 2023 2024; do
      python src/fetch_gamelogs.py $y
    done

Or on Windows:

    for %y in (2015 2016 2017 2018 2019 2021 2022 2023 2024) do python src\fetch_gamelogs.py %y

Download odds workbooks manually into `data/raw/odds/` (see `data/README.md`),
then build and evaluate:

    python src/build_spine.py
    python src/build_features.py
    python src/build_odds.py
    python src/join_odds.py
    python src/baseline_elo.py
    python src/train_model.py
    python src/evaluate_market.py
    python src/find_edges.py

Data files are not committed. Retrosheet downloads are scripted; odds workbooks
are not.

## Attribution

The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark,
DE 19711.
