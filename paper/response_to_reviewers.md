# Response to Reviewers

## Reviewer 1

We thank the reviewer for a careful reading. The comments led to new analyses (a life-table lookup
baseline, per-age-group errors, bootstrap intervals, a group-aware permutation test and a threshold
sensitivity analysis) and to a clearer interpretation of the audit. They also exposed an error in our
original permutation p-values for age-group cross-validation, which we have corrected. All new
numbers are produced by `mlforle12.py` (Part 2c) and saved in `outputs/tables/review_*.csv`.

---

**Comment 1. Labels depend entirely on age, so they cannot capture individual lifestyle effects.
Interpret the negative results as a limitation of the labelling scheme.**

We agree and have reframed the result throughout. All participants in an age group share one label,
so lifestyle can predict the label only through its correlation with age. The weak performance of
models without age and its proxies is therefore a property of the labelling scheme, not evidence
that lifestyle does not affect longevity.

*Changes:* abstract; new closing paragraph of Sect. 4.3; Discussion, which now notes that lifestyle
effects are reported elsewhere [Jackowska 2024; cross-country studies] and that feature effects
require a target that varies between individuals (e.g. observed survival from the cyber-physical
system).

---

**Comment 2. Age is a legitimate predictor of remaining life expectancy. Please distinguish circular
target construction from training–test leakage.**

We agree that our earlier use of "leakage" conflated two issues, and Sect. 3 now separates them:

- **Circular target construction.** The label is a step function of age, so any model with access to
  age or its proxies recovers the life-table lookup. Age is a valid predictor; the problem is that it
  fully determines the label by construction.
- **Validation design.** Random folds place participants of the same age group (who share one label)
  in both training and test sets, so models interpolate between known labels. Leave-one-age-group-out
  CV tests prediction at unseen ages.

We also state that there is no train–test contamination in the usual sense. Augmentation is applied
to training rows only, and no fitted statistic crosses folds except the median imputation of a
single missing muscle-mass value.

*Changes:* the "Leakage audit" is renamed "Target and validation audit" (Sect. 3, Sect. 4.3,
Fig. 1); the Introduction rewords "target leakage" as circularity; scenario A is no longer labelled
"leaky".

---

**Comment 3. Include direct life-table lookup as a baseline.**

Added. The lookup reproduces the labels exactly (MAE 0.00, RMSE 0.00, R² 1.00) without training. It
is now the first row of Table 1, and Sect. 4.2 states that the learned models only approximate a
function known in closed form and add no information beyond it.

---

**Comment 4. Report group sizes and group-specific errors for age-group hold-out.**

Added as a new Table 3 (group size, label and MAE per held-out age group for scenarios B and E,
plus macro averages). The pooled errors are dominated by the 15–19 group (63 of 95 participants);
when it is held out, only 32 older participants remain for training. Macro-averaged over the nine
groups, GB errs by 4.6 years in B against 17.4 for the baseline. In E, all models lie within 0.1
years of the baseline (17.3–17.5). We also describe age-group CV explicitly as a test of prediction
at unseen ages.

---

**Comment 5. Explain the permutation procedure, report uncertainty around baseline improvements,
and justify and test the 0.2 threshold.**

- **Permutation procedure** (Sect. 3): each model is refit on 100 permuted targets with the same
  folds, and p = (1 + #{MAE_perm ≤ MAE_obs}) / 101, so the minimum is 0.01.
- **Correction.** For age-group CV, our original test shuffled labels across all participants. That
  breaks the constant-label structure of the held-out groups and gives an easier null, so the
  reported p-values (0.69 / 0.13 in scenario B) were misleading. We now permute which label each age
  group carries. Under this null, GB in scenario B is significant (p = 0.01) and RF is not
  (p = 0.17); in scenario E neither is (p = 0.56 / 0.54). The earlier statement that "the gain in B
  is not significant" has been corrected.
- **Uncertainty.** Table 2 now reports 95% intervals for the gain over the mean baseline, from a
  paired bootstrap over participants (2000 resamples). In scenarios D and E, every interval includes
  zero.
- **Threshold.** 0.2 lies in the largest gap of the correlation profile: BMI ρ = 0.35 and body fat
  0.24, against at most 0.14 for all other features. Sensitivity results are reported in Sect. 4.3
  (full table in `review_proxy_threshold_sensitivity.csv`):

  | Threshold | Removed | RF gain, random / group CV | GB gain, random / group CV |
  |---|---|---|---|
  | 0.1 | BMI, body fat, sleep, body water, muscle, chronic | −3% / −4% | −11% / −16% |
  | 0.2 | BMI, body fat | −2% / −1% | −12% / −3% |
  | 0.3 | BMI | 7% / 1% (CIs include 0) | 4% / 4% (CIs include 0) |
  | 0.4 | none | 26% / 22% | 35% / 36% |

  The conclusion is unchanged unless BMI, the dominant proxy, is retained.

---

**Comment 6. Relabelling the same participants for 192 countries does not establish population
generalisation.**

We agree and have limited the claims. The step is now called "re-labelling" rather than "transfer"
(Fig. 1, Sect. 3, Sect. 4.4) and is described as a counterfactual: what this cohort's labels would
be under another mortality schedule. Sect. 4.4 states that this shows the audit's conclusions hold
under any schedule, but not that the models generalise to other populations, whose risk-factor
distributions and their relation to survival may differ. The "Generalisability" design goal and the
Discussion say the same.

---

**Comment 7. Explain the different errors in Tables 1 and 2, specify augmentation settings, and
correct the unresolved section reference.**

- **Tables 1 vs 2.** Table 1 uses the unregularised models of the original study and averages 25 fold
  MAEs (5×5 repeated CV). Table 2 uses regularised models (depth ≤ 4, ≥ 5 samples per leaf) with
  pooled out-of-fold predictions from one 5-fold split, the same split the permutation test uses.
  Hence RF 1.87 and GB 0.99 in scenario A against 0.66 and 0.52 in Table 1. This is stated at the
  start of Sect. 4.3.
- **Augmentation.** Two noisy copies of the 76 training rows. The noise is Gaussian with SD equal to
  1% of each feature's training SD, added to the seven continuous features (age, BMI, sleep, steps,
  body water, muscle mass, body fat). Labels are copied unchanged, and the 19 test rows stay clean
  (Sect. 3).
- **Section reference.** The reference in the Design goals paragraph resolves to Sect. 4.3 in our
  build, and the revised PDF has no unresolved references. An unresolved reference ("??") appears
  when LaTeX is run only once, so the camera-ready PDF is compiled with two passes.

---

## Reviewer 2

We thank the reviewer for these comments. They led us to replace the two-year hold-out with a full
rolling-origin evaluation, add feature ablations and a data-quality screen, and correct our earlier
2024 projection. New results come from `mlforle12.py` (Part 4b) and are saved in
`outputs/tables/forecast_rolling_*.csv`, `forecast_feature_ablation.csv`,
`forecast_flagged_series_errors.csv` and `data_quality_flags.csv`.

---

**Comment 1. The forecasting model appears independent of the cohort's lifestyle data; clarify the
connection between the two experiments.**

The reviewer is right that the forecaster uses no cohort data, and Sect. 3 now says so explicitly.
The framework has two components that share only the life tables. The *cohort component* (steps 1–5)
labels and models person-level data. The *life-table component* (step 6) forecasts the tables from
their own history. Its role is to supply life tables for years not yet published, which the labelling
step can use to label a cohort for a future year. Fig. 1 and the Discussion were updated accordingly.
Given the results below, the Discussion concludes that the latest published table (persistence) is
currently the best basis for next-year labels.

---

**Comment 2. Two recovery years provide limited evidence; evaluate all feasible rolling forecast
origins and select models using earlier years.**

Done. Because the models use three lags and the tables start in 2015, the first year with full
features is 2018. Every year from 2019 to 2023 is therefore a forecast origin, each trained only on
earlier years. The new Table 4 reports MAE per origin and pooled MAE, RMSE and 95th-percentile error:

| Model | 2019 | 2020 | 2021 | 2022 | 2023 | Pooled MAE | Pooled RMSE |
|---|---|---|---|---|---|---|---|
| Persistence | 0.19 | **0.85** | 0.88 | 1.46 | **0.56** | **0.79** | **1.21** |
| Linear trend | **0.17** | 0.95 | 1.11 | **1.09** | 1.06 | 0.88 | 1.27 |
| Ridge | 0.18 | 0.93 | **0.83** | 1.77 | 0.76 | 0.89 | 1.35 |
| Random Forest | **0.17** | 0.93 | 0.97 | 1.82 | 0.78 | 0.93 | 1.43 |
| Gradient Boosting | **0.17** | 0.93 | 0.94 | 1.77 | 0.79 | 0.92 | 1.40 |

We also added a *selected* forecaster that, at each origin, uses the model with the lowest mean
error at all earlier origins. It averages MAE 1.09 over 2020–2023, against 0.94 for always using
persistence: the best past model kept changing across the COVID-19 regimes. We note in the
Limitations that only five origins exist, four in or after the COVID-19 shock.

---

**Comment 3. Describe forecastability categories as retrospective error bands unless independently
validated.**

Agreed. The appendix column is renamed "Band" and defined as a *retrospective error band* of the
rolling-origin MAE, not an independently validated measure of forecastability. It is now based on
all five origins rather than the 2023 hold-out alone.

---

**Comment 4. Persistence has the lowest 2023 MAE while GB has the lowest RMSE; discuss this trade-off
before identifying an overall winner.**

We investigated the trade-off and found it was produced by four unstable source series (Comment 5).
On all 192 countries, persistence had the lowest 2023 MAE (0.69) but GB the lowest RMSE (1.66 vs
1.85). Persistence carries the Central African Republic's 2022 value (18.9 years) into 2023, when the
published value is 57.4. That single 38-year error dominates the RMSE. On the 188 stable series,
persistence has the lowest 2023 MAE *and* RMSE (0.56 and 0.80, vs 0.79 and 1.04 for GB), and the
lowest pooled MAE, RMSE and 95th-percentile error over all origins. We therefore identify persistence
as the overall winner only after this analysis, and only for the stable series (Sect. 4.4,
"MAE versus RMSE"). On the four unstable series no model is reliable (MAE 2–7 years at every origin).

---

**Comment 5. Investigate the Central African Republic's projected decline from 57.4 to 35.3 years by
checking its historical series, data processing and model outputs.**

- **Historical series.** The published e⁰ series for 2015–2023 is 51.9, 51.0, 45.1, 52.2, 31.5, 50.6,
  40.1, 18.9, 57.4. It reflects source-level death probabilities (e.g. q for ages 1–4 of 0.24 in 2022).
  These swings are implausible as year-to-year demographic change.
- **Data processing.** Our pipeline passes q, l and e unchanged; preprocessing only parses the age
  group and builds lagged panels. The swings are in the source file.
- **Model outputs.** The 35.3-year projection came from the RF forecaster, which turned the +38.5-year
  jump into 2023 into a reversal through its momentum features.
- **Action.** We screened all series for single-year changes in e⁰ above 10 years. Four are flagged:
  the Central African Republic (38.5), the Holy See (32.5), South Sudan (17.2) and the State of
  Palestine (11.6). The next-largest is Armenia at 6.4, a 2020 drop with known causes. Flagged series
  are excluded from training and evaluation, reported separately, and marked † in the appendix with
  no forecast status. The 2024 projection column has been removed, since persistence, the best
  forecaster, would simply repeat 2023. For Palestine the 2023 drop is plausibly real (conflict
  mortality); the flag means "unstable for forecasting", not "erroneous".

---

**Comment 6. SHAP importance alone does not explain forecasting failure; add feature ablations.**

Added. Removing one feature group at a time, with mean MAE over the five origins:

| Feature set | GB | RF |
|---|---|---|
| Full | 0.92 | 0.93 |
| No momentum (Δ, Δe⁰) | 0.89 | 0.95 |
| Level only (age group, e(t−1)) | **0.84** | **0.87** |
| No country e⁰ | 0.96 | 0.97 |
| No deep lags | 0.93 | 0.93 |
| No log q | 0.93 | 0.93 |

The momentum features that SHAP ranks highest actively hurt: a level-only model improves GB most in
2021–2023 (2023: 0.79 → 0.59). Country e⁰ helps, and deep lags and mortality have no effect. The
learned models thus fail by extrapolating recent changes across regime shifts, and even the
level-only model does not beat persistence (0.84 vs 0.79).

---

**Comment 7. Provide the life-table source, code, preprocessing details and cohort ethics
information.**

- **Preprocessing** is now described in Sect. 3. Life tables are used as published (age-group parsing
  and lagged panels only). Cohort cleaning: decimal format, trailing spaces, ordinal activity coding,
  and median imputation of one missing value.
- **Life-table source:** [TODO: source and release].
- **Code:** [TODO: repository URL].
- **Ethics:** [TODO: approval body and reference number, informed-consent procedure, data
  availability].
