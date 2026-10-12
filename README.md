# From One Country to 192: Life Expectancy Prediction from Single-Country Life-Factor Data

Code for the paper **"An Experimental Framework for Life Expectancy Prediction from Single-Country
Life-Factor Data"** by Asina Bopushova (Ala-Too International University, Bishkek, Kyrgyzstan) and
Ruslan Isaev (Westminster International University in Tashkent, Uzbekistan), presented at ACISM 2026,
the 2nd International Conference on Applied Computational Intelligence and Systems Modeling
(October 16–17, 2026).

The framework takes person-wise life-factor data from one country (here, a Kyrgyz cohort collected
with a cyber-physical system) and links it to the abridged life tables of 192 countries. It has two
components that share only the life tables:

- **Cohort component.** Labels each participant with the remaining life expectancy of their age
  group, trains a mean baseline, a direct life-table lookup, Ridge, Random Forest and Gradient
  Boosting, and audits the result. The audit separates *circular target construction* (the label
  is built from age) from *validation design* (random folds mix age groups).
- **Life-table component.** Forecasts the life tables themselves at every feasible rolling origin
  (2019–2023), against persistence and linear-trend baselines, with feature ablations and a
  data-quality screen for unstable series.

## Main findings

- The cohort models reach a cross-validated R² of 0.96, but a direct life-table lookup is exact, and
  SHAP attributes 97–98% of the prediction to age.
- Without age and its proxies (BMI, body fat), Random Forest and Gradient Boosting do not beat the
  mean baseline. This is a limitation of life-table labels, not evidence against lifestyle effects.
- Re-labelling the same cohort with 192 life tables is a counterfactual, not population
  generalisation.
- For life-table forecasting, persistence has the lowest pooled error (MAE 0.79 years). Feature
  ablations show that the learned models fail by extrapolating recent changes.

## Repository structure

```
mlforle12.py                 Full analysis pipeline (all tables and figures)
mlforle12_original_colab.py  Original Colab notebook of the earlier study, kept for reference
make_paper_assets.py         Builds the paper's appendix table from outputs/tables
life_table_compact.csv       Abridged life tables, 192 countries, 2015–2023 (UN WPP 2024)
data.csv                     Cohort data (95 participants, 30 variables)
requirements.txt             Pinned Python dependencies
outputs/
  tables/                    All result tables (CSV)
  figures/                   Summary figures
  countries/                 One figure per country (192)
  models/                    Saved models and feature lists
paper/                       LNNS paper (main.tex) and response to reviewers
conference/                  ACISM 2026 slides (slides.tex) and slide assets
```

## Data

- **Life tables.** United Nations, Department of Economic and Social Affairs, Population Division
  (2024). *World Population Prospects 2024*. https://population.un.org/wpp/ — abridged life tables
  (q<sub>x</sub>, l<sub>x</sub>, e<sub>x</sub>) for 19 age groups, used as published.
- **Cohort.** 95 participants from Kyrgyzstan, with smart-scale body composition and self-reported
  lifestyle data, collected with the cyber-physical system described in Bopushova & Isaev (2026).
  The data are anonymised: no names or contact details were recorded.
  <!-- TODO: data-availability statement (public in this repository / on request) -->

## Installation

Python 3.12 is required.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Alternatively, with [uv](https://docs.astral.sh/uv/) and no manual environment:

```bash
uv run --python 3.12 --with-requirements requirements.txt python mlforle12.py
```

## Running the analysis

```bash
python mlforle12.py
```

This regenerates everything in `outputs/`. A full run takes about 20–30 minutes on a laptop,
mostly for the permutation tests and the per-country models. All random seeds are fixed (42).

The pipeline runs these parts in order:

| Part | What it does | Key outputs (`outputs/tables/`) |
|---|---|---|
| 1 | Kyrgyzstan life table, compared with the original study's table | `kyrgyzstan_*`, `original_table_vs_kyrgyzstan.csv` |
| 2 | Cohort models: hold-out, 5×5 CV, train-only augmentation, age ablation, SHAP | `participants_*` |
| 2b | Scenarios A–E: no age, no proxies, age-group CV, permutation tests | `leakage_free_results.csv`, `age_proxy_correlations.csv` |
| 2c | Lookup baseline, bootstrap intervals, group-aware permutation, per-group errors, threshold sensitivity | `review_*` |
| 3 | Cohort re-labelled with all 192 life tables | `cohort_by_country.csv` |
| 4 | Two-year forecasting hold-out (2022, 2023) | `forecast_*` |
| 4b | Rolling origins 2019–2023, model selection, ablations, data-quality flags | `forecast_rolling_*`, `forecast_feature_ablation.csv`, `data_quality_flags.csv` |
| 5 | Country summary and one figure per country | `country_summary.csv`, `outputs/countries/` |
| 6 | SHAP feature importance for every country | `country_feature_importance.csv` |

A short text summary is written to `outputs/insights.md`.

## Building the paper and slides

```bash
python make_paper_assets.py          # writes paper/appendix_countries.tex
cd paper && pdflatex main && pdflatex main
```

The paper uses Springer's `llncs.cls` (on CTAN, TeX Live and Overleaf).

```bash
python conference/make_slide_assets.py
cd conference && pdflatex slides && pdflatex slides
```

## Citation

If you use this code, please cite the paper and the original data-collection study:

```bibtex
@inproceedings{bopushova2026framework,
  author    = {Bopushova, Asina and Isaev, Ruslan},
  title     = {An Experimental Framework for Life Expectancy Prediction from Single-Country Life-Factor Data},
  booktitle = {Applied Computational Intelligence and Systems Modeling (ACISM 2026)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026},
  note      = {to appear}
}

@inproceedings{bopushova2026cps,
  author    = {Bopushova, Asina and Isaev, Ruslan},
  title     = {Design and implementation of cyber-physical data collection system of risk factors
               mitigation and life expectancy prediction},
  booktitle = {2026 International Conference on Computer Systems and Technologies (CompSysTech)},
  pages     = {1--6},
  publisher = {IEEE},
  address   = {Ruse, Bulgaria},
  year      = {2026},
  doi       = {10.1109/CompSysTech69798.2026.11683147}
}
```

## License

<!-- TODO: choose a license, e.g. MIT for the code; the life tables follow the UN WPP terms of use -->
