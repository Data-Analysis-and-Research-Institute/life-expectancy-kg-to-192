# -*- coding: utf-8 -*-
"""Life-expectancy modelling with Kyrgyzstan as the reference country.

Fixed, locally runnable version of MlforLE12.ipynb (originally a Colab notebook;
the untouched original is kept in mlforle12_original_colab.py).

Parts
  1. Kyrgyzstan life table - the starting point of the original study
     (replaces the hard-coded life_table dict with WHO-style data from
     life_table_compact.csv).
  2. Participant models (data.csv): Base_LE from the Kyrgyzstan 2023 life table,
     one baseline (mean predictor) + Ridge + Random Forest + Gradient Boosting,
     hold-out + repeated k-fold CV, train-only augmentation, age ablation,
     feature importance, comparison with the paper.
  2b. Leakage-free RF and GB: Base_LE is a lookup of age, so age (and its
     proxies) are removed and/or whole age groups are held out in CV;
     permutation tests check whether any remaining skill is real.
  3. The same cohort re-labelled with every country's 2023 life table.
  4. Forecasting life expectancy for every country and age group
     (hold-out years 2022 and 2023): naive baseline, linear trend, Ridge,
     Random Forest, Gradient Boosting + a 2024 projection.
  5. Per-country tables and plots, cross-country comparison, insights.
  6. Feature importance (SHAP) for every country: full, leakage-free and
     forecasting models.

Run (no system Python needed, uv creates an isolated environment):
  uv run --python 3.12 --with pandas --with scikit-learn --with matplotlib --with joblib --with shap python mlforle12.py
Everything is written to ./outputs
"""

import re
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
    permutation_test_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
LIFE_TABLE_CSV = BASE / "life_table_compact.csv"
PARTICIPANTS_CSV = BASE / "data.csv"

OUT = BASE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
COUNTRY_FIG = OUT / "countries"
MODELS_DIR = OUT / "models"
for d in (FIG, TAB, COUNTRY_FIG, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

REF_COUNTRY = "Kyrgyzstan"  # starting point of the original study
REF_YEAR = 2023  # latest year in the life table
PEERS = ["Kazakhstan", "Tajikistan", "Uzbekistan", "Turkmenistan", "Russian Federation"]
HOLDOUT_YEARS = [2022, 2023]
SEED = 42
AGE_PROXY_THRESHOLD = 0.2  # |Spearman rho| with age above which a feature counts as an age proxy
N_PERMUTATIONS = 100

# Features used in the original study
FEATURES = [
    "age",
    "sex",
    "BMI",
    "sleep_hours",
    "avg_steps_per_day",
    "physical_activity",
    "smoking_status",
    "alcohol_units_per_week",
    "body_water_kg",
    "muscle_mass_kg",
    "body_fat_percent",
    "chronic_disease",
]

# Results reported in the original paper (for comparison only)
PAPER_RESULTS = pd.DataFrame(
    {
        "Model": ["Random Forest", "Gradient Boosting", "Ridge"],
        "MAE_paper": [0.39, 0.18, 0.95],
        "RMSE_paper": [1.15, 0.47, 1.10],
        "R2_paper": [0.99, 1.00, 0.99],
    }
)

# Hard-coded life table used in the original notebook (kept for comparison)
ORIGINAL_STUDY_TABLE = {
    0: 74.6, 5: 70.7, 10: 65.8, 15: 60.9, 20: 56.1, 25: 51.3, 30: 46.5, 35: 41.7,
    40: 37.0, 45: 32.4, 50: 28.0, 55: 23.7, 60: 19.6, 65: 15.7, 70: 12.1,
    75: 8.8, 80: 5.5, 85: 3.4,
}

# ---------------------------------------------------------------------------
# Plot style (validated reference palette; text stays in ink colours)
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
YEAR_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]  # sequential blue, old -> new
COUNTRY_COLOR, REF_COLOR = SERIES[0], SERIES[1]
MODEL_COLORS = {
    "Baseline": MUTED,
    "Linear trend": SERIES[3],
    "Ridge": SERIES[2],
    "Random Forest": SERIES[4],
    "Gradient Boosting": SERIES[6],
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)


def spread(values, min_gap):
    """Nudge label positions apart so direct labels do not collide."""
    order = np.argsort(values)
    out = np.asarray(values, dtype=float).copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


LABEL_BOX = {"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_life_table(path=LIFE_TABLE_CSV):
    lt = pd.read_csv(path).rename(
        columns={
            "ISO3": "iso3",
            "Country": "country",
            "Year": "year",
            "Age (x)": "age_group",
            "Death prob. qx": "qx",
            "Survivors lx": "lx",
            "Life exp. ex": "ex",
        }
    )
    lt["age_group"] = lt["age_group"].astype(str)
    lt["age_start"] = lt["age_group"].str.extract(r"^(\d+)")[0].astype(int)
    return lt.sort_values(["country", "year", "age_start"]).reset_index(drop=True)


ACTIVITY = {"sedentary": 0, "low": 1, "moderate": 2, "high": 3}
INCOME = {"low": 0, "medium": 1, "moderate": 1, "high": 2}
HAZARD = {"low": 0, "medium": 1, "high": 2}
BODY_TYPE = {"underweight": 0, "slim": 1, "lean": 2, "fit": 3, "overweight": 4, "obese": 5}


def load_participants(path=PARTICIPANTS_CSV):
    # data.csv uses ';' as separator and ',' as decimal mark
    df = pd.read_csv(path, sep=";", decimal=",")

    # Trailing spaces ("moderate ", "urban ") created fake categories before
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().str.lower()

    # Ordinal encodings keep the natural order (LabelEncoder sorted alphabetically)
    df["physical_activity"] = df["physical_activity"].map(ACTIVITY)
    df["income_quintile"] = df["income_quintile"].map(INCOME)
    df["occupational_hazard"] = df["occupational_hazard"].map(HAZARD)
    df["urban_rural"] = (df["urban_rural"] == "rural").astype(int)
    df["immigrant_status"] = (df["immigrant_status"] == "immigrant").astype(int)
    df["body_type"] = df["body_type"].map(BODY_TYPE)
    if not pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = df["sex"].map({"male": 0, "female": 1})

    # Any remaining text columns -> numeric (handles stray decimal commas)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype("string").str.replace(",", "."), errors="coerce")
        df[col] = df[col].astype(float)

    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing):
        print("Missing values imputed with the median:", n_missing.to_dict())
    return df.fillna(df.median(numeric_only=True))


def life_table_for(lt, country, year=REF_YEAR):
    t = lt[(lt["country"] == country) & (lt["year"] == year)]
    return t.set_index("age_start")["ex"].sort_index()


def base_le(ages, table):
    """Remaining life expectancy e(x) of the age group each age falls into."""
    starts = table.index.to_numpy()
    idx = np.searchsorted(starts, np.asarray(ages, dtype=float), side="right") - 1
    return table.to_numpy()[idx]


# ---------------------------------------------------------------------------
# Part 1 - Kyrgyzstan life table (starting point)
# ---------------------------------------------------------------------------
def part1_kyrgyzstan(lt):
    print("\n=== Part 1: Kyrgyzstan life table ===")
    kg = lt[lt["country"] == REF_COUNTRY]
    wide = kg.pivot(index=["age_start", "age_group"], columns="year", values="ex").round(2)
    wide.to_csv(TAB / "kyrgyzstan_life_expectancy_by_age_year.csv")

    ref = kg[kg["year"] == REF_YEAR].set_index("age_start")
    comp = pd.DataFrame(
        {
            "age_group": ref["age_group"],
            "original_study_table": pd.Series(ORIGINAL_STUDY_TABLE),
            f"kyrgyzstan_{REF_YEAR}": ref["ex"].round(2),
        }
    ).dropna(subset=["age_group"])
    comp["difference"] = (comp[f"kyrgyzstan_{REF_YEAR}"] - comp["original_study_table"]).round(2)
    comp.to_csv(TAB / "original_table_vs_kyrgyzstan.csv")
    print(comp.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for color, year in zip(YEAR_RAMP, [2015, 2019, 2021, 2023]):
        s = kg[kg["year"] == year]
        ax.plot(s["age_start"], s["ex"], color=color, label=str(year))
    ax.plot(list(ORIGINAL_STUDY_TABLE), list(ORIGINAL_STUDY_TABLE.values()),
            color=MUTED, linestyle="--", label="Original study table")
    ax.set(title="Kyrgyzstan: remaining life expectancy e(x)", xlabel="Age (start of group)",
           ylabel="Years remaining")
    ax.legend()

    ax = axes[1]
    e0 = kg[kg["age_start"] == 0]
    ax.plot(e0["year"], e0["ex"], color=COUNTRY_COLOR, marker="o", markersize=5)
    for _, r in e0.iterrows():
        ax.annotate(f"{r['ex']:.1f}", (r["year"], r["ex"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set(title="Kyrgyzstan: life expectancy at birth", xlabel="Year", ylabel="e0 (years)")

    ax = axes[2]
    for color, year in zip([YEAR_RAMP[0], YEAR_RAMP[2], YEAR_RAMP[3]], [2019, 2021, 2023]):
        s = kg[kg["year"] == year]
        ax.plot(s["age_start"], s["qx"], color=color, label=str(year))
    ax.set_yscale("log")
    ax.set(title="Kyrgyzstan: probability of dying qx (log)", xlabel="Age (start of group)", ylabel="qx")
    ax.legend()
    savefig(fig, FIG / "01_kyrgyzstan_life_table.png")
    return comp


# ---------------------------------------------------------------------------
# Part 2 - participant models with Kyrgyzstan as reference
# ---------------------------------------------------------------------------
def participant_models(n_trees=200):
    return {
        "Baseline": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                       random_state=SEED),
    }


def augment(X, y, copies=2, noise=0.01, seed=SEED):
    """Gaussian-noise copies of the TRAINING rows only (continuous columns)."""
    rng = np.random.default_rng(seed)
    cont = [c for c in X.columns if X[c].nunique() > 5]
    Xs, ys = [X], [y]
    for _ in range(copies):
        Xn = X.copy()
        Xn[cont] = Xn[cont] + rng.normal(0, noise * X[cont].std().to_numpy(), size=Xn[cont].shape)
        Xs.append(Xn)
        ys.append(y)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def cv_scores(models, X, y, cv):
    rows = []
    for name, model in models.items():
        s = cross_validate(clone(model), X, y, cv=cv, n_jobs=-1,
                           scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
        rows.append(
            {
                "Model": name,
                "CV_MAE": -s["test_neg_mean_absolute_error"].mean(),
                "CV_MAE_sd": s["test_neg_mean_absolute_error"].std(),
                "CV_RMSE": -s["test_neg_root_mean_squared_error"].mean(),
                "CV_R2": s["test_r2"].mean(),
                "CV_R2_sd": s["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows)


def part2_participants(df, lt):
    print("\n=== Part 2: participant models (Kyrgyzstan reference) ===")
    kg_table = life_table_for(lt, REF_COUNTRY)
    df = df.copy()
    df["Base_LE"] = base_le(df["age"], kg_table)
    X, y = df[FEATURES], df["Base_LE"].to_numpy()
    print(f"X shape {X.shape}, unique Base_LE values: {len(np.unique(y))}")

    models = participant_models()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_aug, y_aug = augment(X_train, y_train)

    rows, preds = [], {}
    for name, model in models.items():
        m_real = clone(model).fit(X_train, y_train)
        m_aug = clone(model).fit(X_aug, y_aug)
        preds[name] = m_real.predict(X_test)
        r = {"Model": name}
        r.update({f"{k}_real": v for k, v in metrics(y_test, preds[name]).items()})
        r.update({f"{k}_aug": v for k, v in metrics(y_test, m_aug.predict(X_test)).items()})
        rows.append(r)
    holdout = pd.DataFrame(rows)

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv = cv_scores(models, X, y, rkf)
    results = holdout.merge(cv, on="Model").merge(PAPER_RESULTS, on="Model", how="left")
    results.round(3).to_csv(TAB / "participants_model_comparison.csv", index=False)
    print(results.round(3).to_string(index=False))

    # Age ablation: the target is a function of age, so this shows leakage
    abl = []
    for label, cols in [
        ("All 12 features", FEATURES),
        ("Without age", [c for c in FEATURES if c != "age"]),
        ("Age only", ["age"]),
    ]:
        s = cv_scores(models, df[cols], y, rkf)
        s.insert(0, "Feature set", label)
        abl.append(s)
    ablation = pd.concat(abl, ignore_index=True)
    ablation.round(3).to_csv(TAB / "participants_age_ablation.csv", index=False)
    print(ablation[["Feature set", "Model", "CV_MAE", "CV_R2"]].round(3).to_string(index=False))

    # Comparison table for the first participants (as in the original notebook)
    comparison = pd.DataFrame({"Age": X_test["age"].to_numpy(), "Base_LE": y_test})
    for name, p in preds.items():
        comparison[name] = p.round(2)
    comparison.to_csv(TAB / "participants_test_predictions.csv", index=False)

    # Figure: predicted vs life-table baseline, one panel per model
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    lo, hi = y.min() - 2, y.max() + 2
    for ax, (name, p) in zip(axes, preds.items()):
        ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
        ax.scatter(y_test, p, s=40, color=MODEL_COLORS[name], edgecolor=SURFACE, linewidth=1.5)
        mae = holdout.loc[holdout["Model"] == name, "MAE_real"].iloc[0]
        ax.set(title=f"{name}  (MAE {mae:.2f})", xlabel="Kyrgyzstan life-table e(x)", xlim=(lo, hi),
               ylim=(lo, hi))
    axes[0].set_ylabel("Predicted remaining LE")
    savefig(fig, FIG / "02_participants_pred_vs_actual.png")

    # Figure: model comparison (CV MAE) incl. augmentation + paper
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    names = list(models)
    ax.barh(names[::-1], cv.set_index("Model").loc[names[::-1], "CV_MAE"],
            color=[MODEL_COLORS[n] for n in names[::-1]], height=0.6,
            xerr=cv.set_index("Model").loc[names[::-1], "CV_MAE_sd"], ecolor=AXIS)
    for i, n in enumerate(names[::-1]):
        v, sd = cv.set_index("Model").loc[n, ["CV_MAE", "CV_MAE_sd"]]
        ax.annotate(f"{v:.2f} +/- {sd:.2f}", (v + sd, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set(title="5x5 repeated CV MAE (years), lower is better", xlabel="MAE")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ml = [n for n in names if n != "Baseline"]
    h = holdout.set_index("Model").loc[ml]
    p = PAPER_RESULTS.set_index("Model").loc[ml]
    x = np.arange(len(ml))
    w = 0.26
    ax.bar(x - w, h["MAE_real"], w, color=SERIES[0], label="Real data (hold-out)")
    ax.bar(x, h["MAE_aug"], w, color=SERIES[1], label="Augmented train (hold-out)")
    ax.bar(x + w, p["MAE_paper"], w, color=SERIES[2], label="Paper")
    ax.set_xticks(x, ml)
    ax.set(title="Hold-out MAE: real vs augmented vs paper", ylabel="MAE (years)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    savefig(fig, FIG / "03_participants_model_comparison.png")

    # Feature importance: RF impurity, GB permutation, lifestyle-only view
    rf = clone(models["Random Forest"]).fit(X_train, y_train)
    gb = clone(models["Gradient Boosting"]).fit(X_train, y_train)
    perm = permutation_importance(gb, X_test, y_test, n_repeats=30, random_state=SEED,
                                  scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(
        {"feature": FEATURES, "rf_impurity": rf.feature_importances_,
         "gb_permutation_mae": perm.importances_mean}
    ).sort_values("rf_impurity", ascending=False)
    imp.round(4).to_csv(TAB / "participants_feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    s = imp.sort_values("rf_impurity")
    axes[0].barh(s["feature"], s["rf_impurity"], color=MODEL_COLORS["Random Forest"], height=0.6)
    axes[0].set(title="Random Forest impurity importance", xlabel="Importance")
    s = imp.sort_values("gb_permutation_mae")
    axes[1].barh(s["feature"], s["gb_permutation_mae"], color=MODEL_COLORS["Gradient Boosting"], height=0.6)
    axes[1].set(title="GB permutation importance (MAE increase)", xlabel="Years")
    s = imp[~imp["feature"].isin(["age", "BMI"])].sort_values("rf_impurity")
    axes[2].barh(s["feature"], s["rf_impurity"], color=MUTED, height=0.6)
    axes[2].set(title="RF importance excluding age and BMI", xlabel="Importance")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    savefig(fig, FIG / "04_participants_feature_importance.png")

    tree_models = {n: m for n, m in models.items() if n in ("Random Forest", "Gradient Boosting")}
    shap_imp = shap_analysis(tree_models, X, y, "04b_shap_leaky_all_features",
                             "Leaky setup (all 12 features)")

    # Save best non-baseline model trained on all data
    best = cv[cv["Model"] != "Baseline"].sort_values("CV_MAE").iloc[0]["Model"]
    final = clone(models[best]).fit(X, y)
    joblib.dump(final, MODELS_DIR / "life_expectancy_model.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "feature_names.pkl")
    print(f"Saved best participant model: {best}")
    return {"results": results, "ablation": ablation, "importance": imp, "best": best,
            "shap": shap_imp}


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------
def shap_analysis(models, X, y, tag, title):
    """TreeSHAP for each model fitted on all rows: beeswarm per model + mean |SHAP| bars."""
    mean_abs = {}
    for name, model in models.items():
        fitted = clone(model).fit(X, y)
        sv = shap.TreeExplainer(fitted).shap_values(X)
        mean_abs[name] = np.abs(sv).mean(axis=0)
        pd.DataFrame(sv, columns=X.columns).round(4).to_csv(
            TAB / f"{tag}_{safe_name(name).lower()}_values.csv", index=False)
        shap.summary_plot(sv, X, show=False, plot_size=(8, 0.35 * X.shape[1] + 1.8))
        plt.title(f"{title}: {name} SHAP values", fontsize=11, fontweight="bold", loc="left")
        plt.gcf().savefig(FIG / f"{tag}_{safe_name(name).lower()}_beeswarm.png", dpi=130,
                          bbox_inches="tight", facecolor=SURFACE)
        plt.close("all")
    imp = pd.DataFrame(mean_abs, index=X.columns)
    imp["share_RF_%"] = 100 * imp["Random Forest"] / imp["Random Forest"].sum()
    imp["share_GB_%"] = 100 * imp["Gradient Boosting"] / imp["Gradient Boosting"].sum()
    imp = imp.sort_values("Gradient Boosting", ascending=False)
    imp.round(4).to_csv(TAB / f"{tag}_mean_abs.csv")

    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 0.33 * X.shape[1] + 1.6))
    for ax, name in zip(np.atleast_1d(axes), models):
        s = imp[name].sort_values()
        ax.barh(s.index, s.values, color=MODEL_COLORS[name], height=0.6)
        for i, v in enumerate(s.values):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set(title=f"{name}: mean |SHAP| (years)", xlabel="Mean absolute SHAP value")
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    savefig(fig, FIG / f"{tag}_mean_abs.png")
    print(f"SHAP ({title}) top features:",
          {n: imp[n].sort_values(ascending=False).head(3).round(2).to_dict() for n in models})
    return imp


# ---------------------------------------------------------------------------
# Part 2b - leakage-free Random Forest and Gradient Boosting
# ---------------------------------------------------------------------------
def leakage_free_models():
    # Regularised for n=95; the same settings are used in every scenario so that
    # differences between scenarios come from the leakage handling only.
    return {
        "Baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                               max_features=0.5, random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                       max_depth=2, min_samples_leaf=5,
                                                       subsample=0.8, random_state=SEED),
    }


def part2b_leakage_free(df, lt):
    """Base_LE is a lookup of age, so a model that sees age (or a CV that mixes the
    same age group into train and test) just memorises the life table.

    Remedies evaluated here:
      * drop age, then also drop age proxies (|Spearman rho| >= AGE_PROXY_THRESHOLD);
      * leave-one-age-group-out CV, so the test age group's LE is never seen in training;
      * permutation test (shuffled target) to check any remaining skill is not chance.
    """
    print("\n=== Part 2b: leakage-free Random Forest and Gradient Boosting ===")
    kg_table = life_table_for(lt, REF_COUNTRY)
    starts = kg_table.index.to_numpy()
    y = base_le(df["age"], kg_table)
    groups = starts[np.searchsorted(starts, df["age"].to_numpy(), side="right") - 1]

    rho = df[FEATURES].corrwith(df["age"], method="spearman").drop("age")
    proxies = rho[rho.abs() >= AGE_PROXY_THRESHOLD].index.tolist()
    no_age = [f for f in FEATURES if f != "age"]
    strict = [f for f in no_age if f not in proxies]
    pd.DataFrame({"spearman_rho_with_age": rho.round(3),
                  "age_proxy": rho.abs() >= AGE_PROXY_THRESHOLD}).to_csv(TAB / "age_proxy_correlations.csv")
    print(f"Age proxies (|rho| >= {AGE_PROXY_THRESHOLD}): {proxies}")
    print("Participants per age group:", pd.Series(groups).value_counts().sort_index().to_dict())

    leaky = "A. Leaky reference: all features, random CV"
    scenarios = [
        (leaky, FEATURES, "random"),
        ("B. All features, leave-one-age-group-out CV", FEATURES, "group"),
        ("C. No age, random CV", no_age, "random"),
        ("D. No age + no age proxies, random CV", strict, "random"),
        ("E. No age + no proxies, age-group CV (strictest)", strict, "group"),
    ]
    models = leakage_free_models()
    rows, oof = [], {}
    for label, cols, scheme in scenarios:
        X = df[cols]
        if scheme == "group":
            splits = list(LeaveOneGroupOut().split(X, y, groups))
        else:
            splits = list(KFold(5, shuffle=True, random_state=SEED).split(X))
        for name, model in models.items():
            if scheme == "random":  # average over 5 different shuffles
                runs = []
                for r in range(5):
                    cv = KFold(5, shuffle=True, random_state=SEED + r)
                    p = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
                    runs.append(metrics(y, p))
                    if r == 0:
                        oof[(label, name)] = p
                m = {k: float(np.mean([s[k] for s in runs])) for k in runs[0]}
            else:
                p = cross_val_predict(clone(model), X, y, cv=splits, n_jobs=-1)
                oof[(label, name)] = p
                m = metrics(y, p)
            row = {"Scenario": label, "n_features": len(cols), "CV": scheme, "Model": name, **m}
            if name != "Baseline" and label != leaky:
                # splits are fixed, the target is shuffled globally (not within age groups)
                _, _, pval = permutation_test_score(
                    clone(model), X, y, cv=splits, n_permutations=N_PERMUTATIONS,
                    scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
                row["permutation_p_value"] = pval
            rows.append(row)
    res = pd.DataFrame(rows)
    base = res[res["Model"] == "Baseline"].set_index("Scenario")["MAE"]
    res["MAE_gain_vs_baseline_%"] = 100 * (1 - res["MAE"] / res["Scenario"].map(base))
    res.round(4).to_csv(TAB / "leakage_free_results.csv", index=False)
    print(res[["Scenario", "Model", "MAE", "R2", "MAE_gain_vs_baseline_%", "permutation_p_value"]]
          .round(3).to_string(index=False))

    # Figure: MAE and gain over baseline per scenario
    labels = [s[0] for s in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    yy = np.arange(len(labels))[::-1]
    h = 0.26
    for j, name in enumerate(models):
        r = res[res["Model"] == name].set_index("Scenario").loc[labels]
        off = (1 - j) * h
        axes[0].barh(yy + off, r["MAE"], h, color=MODEL_COLORS[name], label=name)
        for yi, v in zip(yy + off, r["MAE"]):
            axes[0].annotate(f"{v:.2f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                             va="center", fontsize=7, color=INK2)
        if name != "Baseline":
            axes[1].barh(yy + off, r["MAE_gain_vs_baseline_%"], h, color=MODEL_COLORS[name], label=name)
            for yi, v, pv in zip(yy + off, r["MAE_gain_vs_baseline_%"], r["permutation_p_value"]):
                txt = f"{v:.0f}%" + ("" if pd.isna(pv) else f"  p={pv:.2f}")
                axes[1].annotate(txt, (max(v, 0), yi), xytext=(3, 0), textcoords="offset points",
                                 va="center", fontsize=7, color=INK2)
    axes[0].set_yticks(yy, [textwrap_label(l) for l in labels])
    axes[0].set(title="Out-of-fold MAE (years), lower is better", xlabel="MAE")
    axes[1].axvline(0, color=AXIS, linewidth=1)
    axes[1].set(title="MAE improvement over the baseline (%)", xlabel="% (p = permutation test)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    savefig(fig, FIG / "02b_leakage_scenarios.png")

    # Figure: out-of-fold predictions, leaky vs age-group CV vs strictest
    show = [(leaky, "A. leaky"), (labels[1], "B. age-group CV"), (labels[4], "E. strictest")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    lo, hi = y.min() - 3, y.max() + 3
    for i, name in enumerate(["Random Forest", "Gradient Boosting"]):
        for j, (lab, short) in enumerate(show):
            ax = axes[i, j]
            p = oof[(lab, name)]
            ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
            ax.scatter(y, p, s=30, color=MODEL_COLORS[name], edgecolor=SURFACE, linewidth=1)
            ax.set(title=f"{name}, {short} (MAE {mean_absolute_error(y, p):.2f})",
                   xlim=(lo, hi), ylim=(lo, hi))
            if i == 1:
                ax.set_xlabel("Kyrgyzstan life-table e(x)")
            if j == 0:
                ax.set_ylabel("Out-of-fold prediction")
    savefig(fig, FIG / "02c_leakage_free_pred_vs_actual.png")

    # SHAP: what the models use once age is gone
    tree = {n: m for n, m in models.items() if n != "Baseline"}
    shap_no_age = shap_analysis(tree, df[no_age], y, "04c_shap_no_age", "No age (11 features)")
    shap_strict = shap_analysis(tree, df[strict], y, "04d_shap_no_age_no_proxies",
                                f"No age, no age proxies ({len(strict)} features)")

    final = clone(models["Gradient Boosting"]).fit(df[strict], y)
    joblib.dump(final, MODELS_DIR / "life_expectancy_model_leakage_free.pkl")
    joblib.dump(strict, MODELS_DIR / "feature_names_leakage_free.pkl")
    return {"results": res, "proxies": proxies, "strict": strict, "rho": rho,
            "shap_no_age": shap_no_age, "shap_strict": shap_strict}


def textwrap_label(s, width=30):
    return "\n".join(textwrap.wrap(s, width))


# ---------------------------------------------------------------------------
# Part 2c - reviewer analyses: lookup baseline, per-age-group errors,
#           bootstrap uncertainty, age-proxy threshold sensitivity
# ---------------------------------------------------------------------------
N_BOOT = 2000


def bootstrap_gain(y, p_model, p_base, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap (participants resampled) of the MAE gain over the baseline, in %."""
    rng = np.random.default_rng(seed)
    e_m, e_b = np.abs(y - p_model), np.abs(y - p_base)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    gains = 100 * (1 - e_m[idx].mean(axis=1) / e_b[idx].mean(axis=1))
    return 100 * (1 - e_m.mean() / e_b.mean()), *np.percentile(gains, [2.5, 97.5])


def part2c_review(df, lt):
    print("\n=== Part 2c: reviewer analyses ===")
    kg_table = life_table_for(lt, REF_COUNTRY)
    starts = kg_table.index.to_numpy()
    y = base_le(df["age"], kg_table)
    groups = starts[np.searchsorted(starts, df["age"].to_numpy(), side="right") - 1]
    models = leakage_free_models()
    no_age = [f for f in FEATURES if f != "age"]
    rho = df[FEATURES].corrwith(df["age"], method="spearman").drop("age")
    strict_02 = [f for f in no_age if abs(rho[f]) < AGE_PROXY_THRESHOLD]

    # 1) Direct life-table lookup: reproduces the labels exactly, needs no training
    lookup = base_le(df["age"], kg_table)
    lookup_row = {"Model": "Life-table lookup", **metrics(y, lookup)}
    print("Lookup baseline:", {k: round(v, 4) for k, v in lookup_row.items() if k != "Model"})

    # Augmentation settings actually used in Part 2 (training split only)
    X_train, _, _, _ = train_test_split(df[FEATURES], y, test_size=0.2, random_state=SEED)
    aug_cols = [c for c in X_train.columns if X_train[c].nunique() > 5]
    print("Augmented (continuous) columns:", aug_cols)

    def oof(model, cols, scheme, seed=SEED):
        X = df[cols]
        if scheme == "group":
            cv = list(LeaveOneGroupOut().split(X, y, groups))
        else:
            cv = KFold(5, shuffle=True, random_state=seed)
        return cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)

    scenarios = {
        "A": (FEATURES, "random"), "B": (FEATURES, "group"), "C": (no_age, "random"),
        "D": (strict_02, "random"), "E": (strict_02, "group"),
    }

    # 2) Bootstrap 95% CIs of the gain over the baseline for every scenario
    rows, preds = [], {}
    for sc, (cols, scheme) in scenarios.items():
        p = {n: oof(m, cols, scheme) for n, m in models.items()}
        preds[sc] = p
        for n in ("Random Forest", "Gradient Boosting"):
            g, lo, hi = bootstrap_gain(y, p[n], p["Baseline"])
            rows.append({"Scenario": sc, "Model": n, "MAE": mean_absolute_error(y, p[n]),
                         "Baseline_MAE": mean_absolute_error(y, p["Baseline"]),
                         "gain_%": g, "ci_low_%": lo, "ci_high_%": hi})
    ci = pd.DataFrame(rows)

    # Group-aware permutation test for age-group CV: permute which label each age group
    # carries (labels stay constant within a group), so the null keeps the CV structure.
    # A global shuffle would break that structure and give an easier, misleading null.
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    label_of = {g: y[groups == g][0] for g in uniq}
    perms = [rng.permutation(uniq) for _ in range(N_PERMUTATIONS)]
    for sc in ("B", "E"):
        cols, _ = scenarios[sc]
        splits = list(LeaveOneGroupOut().split(df[cols], y, groups))
        for n in ("Random Forest", "Gradient Boosting"):
            obs = mean_absolute_error(y, preds[sc][n])
            null = []
            for perm in perms:
                mapping = dict(zip(uniq, (label_of[g] for g in perm)))
                y_p = np.array([mapping[g] for g in groups])
                p_ = cross_val_predict(clone(models[n]), df[cols], y_p, cv=splits, n_jobs=-1)
                null.append(mean_absolute_error(y_p, p_))
            pval = (1 + np.sum(np.array(null) <= obs)) / (N_PERMUTATIONS + 1)
            ci.loc[(ci["Scenario"] == sc) & (ci["Model"] == n), "group_perm_p"] = pval
    ci.round(3).to_csv(TAB / "review_bootstrap_gain_ci.csv", index=False)
    print(ci.round(2).to_string(index=False))

    # 3) Group sizes and group-specific errors under leave-one-age-group-out CV
    age_lbl = lt[(lt["country"] == REF_COUNTRY) & (lt["year"] == REF_YEAR)].set_index("age_start")["age_group"]
    g_rows = []
    for g in np.unique(groups):
        m = groups == g
        r = {"age_group": age_lbl[g], "n": int(m.sum()), "label_e(x)": y[m][0]}
        for sc in ("B", "E"):
            for n in models:
                r[f"{sc}_{n}"] = mean_absolute_error(y[m], preds[sc][n][m])
        g_rows.append(r)
    gt = pd.DataFrame(g_rows)
    gt.round(2).to_csv(TAB / "review_group_errors.csv", index=False)
    print(gt.round(2).to_string(index=False))
    macro = gt[[c for c in gt.columns if c.startswith(("B_", "E_"))]].mean()
    print("Macro-average over age groups:", macro.round(2).to_dict())

    # 4) Sensitivity of the age-proxy threshold
    s_rows = []
    for thr in (0.1, 0.2, 0.3, 0.4):
        removed = rho[rho.abs() >= thr].index.tolist()
        cols = [f for f in no_age if f not in removed]
        for scheme in ("random", "group"):
            p = {n: oof(m, cols, scheme) for n, m in models.items()}
            for n in ("Random Forest", "Gradient Boosting"):
                g, lo, hi = bootstrap_gain(y, p[n], p["Baseline"])
                s_rows.append({"threshold": thr, "removed": ", ".join(removed) or "none",
                               "n_features": len(cols), "CV": scheme, "Model": n,
                               "gain_%": g, "ci_low_%": lo, "ci_high_%": hi})
    sens = pd.DataFrame(s_rows)
    sens.round(2).to_csv(TAB / "review_proxy_threshold_sensitivity.csv", index=False)
    print(sens.round(1).to_string(index=False))
    return {"lookup": lookup_row, "ci": ci, "groups": gt, "macro": macro, "sens": sens,
            "aug_cols": aug_cols}


# ---------------------------------------------------------------------------
# Part 3 - the same cohort under every country's life table
# ---------------------------------------------------------------------------
def part3_cohort_by_country(df, lt):
    print("\n=== Part 3: cohort re-labelled with every country's life table ===")
    X = df[FEATURES]
    models = participant_models(n_trees=100)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    iso = lt.drop_duplicates("country").set_index("country")["iso3"]
    rows = []
    countries = sorted(lt["country"].unique())
    for i, country in enumerate(countries, 1):
        yc = base_le(df["age"], life_table_for(lt, country))
        row = {
            "country": country,
            "iso3": iso[country],
            "cohort_mean_remaining_LE": yc.mean(),
            "cohort_mean_age_at_death": (df["age"].to_numpy() + yc).mean(),
        }
        for name, model in models.items():
            p = cross_val_predict(clone(model), X, yc, cv=kf, n_jobs=-1)
            row[f"cohort_cv_MAE_{name}"] = mean_absolute_error(yc, p)
        rows.append(row)
        if i % 40 == 0:
            print(f"  {i}/{len(countries)} countries")
    cohort = pd.DataFrame(rows)
    ref = cohort.loc[cohort["country"] == REF_COUNTRY, "cohort_mean_remaining_LE"].iloc[0]
    cohort["cohort_gap_vs_KGZ"] = cohort["cohort_mean_remaining_LE"] - ref
    ml_cols = [f"cohort_cv_MAE_{n}" for n in models if n != "Baseline"]
    cohort["cohort_best_model"] = (
        cohort[ml_cols].idxmin(axis=1).str.replace("cohort_cv_MAE_", "", regex=False)
    )
    cohort.round(3).to_csv(TAB / "cohort_by_country.csv", index=False)
    print(cohort["cohort_best_model"].value_counts().to_string())

    # Figure: gap vs Kyrgyzstan (top/bottom 12 + peers), diverging blue/red
    s = cohort.sort_values("cohort_gap_vs_KGZ")
    pick = pd.concat([s.head(12), s[s["country"].isin(PEERS + [REF_COUNTRY])], s.tail(12)])
    pick = pick.drop_duplicates("country").sort_values("cohort_gap_vs_KGZ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    colors = [MUTED if c == REF_COUNTRY else (SERIES[0] if v > 0 else SERIES[7])
              for c, v in zip(pick["country"], pick["cohort_gap_vs_KGZ"])]
    ax.barh(pick["country"], pick["cohort_gap_vs_KGZ"], color=colors, height=0.7)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set(title="Same 95 participants: remaining LE vs Kyrgyzstan (years)",
           xlabel="Difference in mean remaining life expectancy")
    ax.grid(axis="y", visible=False)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in PEERS + [REF_COUNTRY]:
            lbl.set_fontweight("bold")

    ax = axes[1]
    data = [cohort[f"cohort_cv_MAE_{n}"] for n in models]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                    medianprops={"color": INK}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for patch, n in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS[n])
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(models) + 1), list(models))
    ax.set_xscale("log")
    ax.set(title="CV MAE across all 192 countries' labels (log scale)", xlabel="MAE (years)")
    ax.grid(axis="y", visible=False)
    savefig(fig, FIG / "05_cohort_by_country.png")
    return cohort


# ---------------------------------------------------------------------------
# Part 4 - forecasting e(x) for every country
# ---------------------------------------------------------------------------
FORECAST_FEATURES = ["age_start", "ex_lag1", "ex_lag2", "ex_lag3", "d1", "d2",
                     "log_qx_lag1", "e0_lag1", "e0_d1"]


def build_panel(lt, extra_year=None):
    p = lt[["iso3", "country", "year", "age_start", "age_group", "qx", "ex"]].copy()
    if extra_year is not None:  # empty rows for a future year (features from lags)
        fut = p[p["year"] == extra_year - 1].copy()
        fut["year"] = extra_year
        fut[["qx", "ex"]] = np.nan
        p = pd.concat([p, fut], ignore_index=True)
    p = p.sort_values(["country", "age_start", "year"]).reset_index(drop=True)
    g = p.groupby(["country", "age_start"])
    for k in (1, 2, 3):
        p[f"ex_lag{k}"] = g["ex"].shift(k)
    p["log_qx_lag1"] = np.log(g["qx"].shift(1).clip(lower=1e-6))
    p["d1"] = p["ex_lag1"] - p["ex_lag2"]
    p["d2"] = p["ex_lag2"] - p["ex_lag3"]
    e0 = p.loc[p["age_start"] == 0, ["country", "year", "ex"]].sort_values(["country", "year"])
    ge = e0.groupby("country")["ex"]
    e0["e0_lag1"] = ge.shift(1)
    e0["e0_d1"] = ge.shift(1) - ge.shift(2)
    p = p.merge(e0[["country", "year", "e0_lag1", "e0_d1"]], on=["country", "year"], how="left")
    p["target_delta"] = p["ex"] - p["ex_lag1"]  # models learn the year-on-year change
    return p


def linear_trend_forecast(lt, target_year):
    keys = ["country", "age_start"]
    h = lt[lt["year"] < target_year]
    agg = h.assign(
        xc=h["year"] - h.groupby(keys)["year"].transform("mean"),
        yc=h["ex"] - h.groupby(keys)["ex"].transform("mean"),
    )
    agg = agg.assign(xy=agg["xc"] * agg["yc"], xx=agg["xc"] ** 2).groupby(keys).agg(
        xy=("xy", "sum"), xx=("xx", "sum"), xm=("year", "mean"), ym=("ex", "mean"))
    return (agg["ym"] + agg["xy"] / agg["xx"] * (target_year - agg["xm"])).rename("Linear trend")


def forecast_models():
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                           random_state=SEED),
    }


def forecast_year(panel, lt, target_year, features=FORECAST_FEATURES, exclude=(), models=None):
    train = panel[(panel["year"] < target_year) & panel["ex_lag3"].notna() & panel["ex"].notna()
                  & ~panel["country"].isin(exclude)]
    test = panel[panel["year"] == target_year].copy()
    test["Baseline"] = test["ex_lag1"]  # naive persistence: next year = last year
    trend = linear_trend_forecast(lt, target_year)
    test = test.merge(trend.reset_index(), on=["country", "age_start"], how="left")
    for name, model in (models or forecast_models()).items():
        m = clone(model).fit(train[features], train["target_delta"])
        test[name] = test["ex_lag1"] + m.predict(test[features])
    return test


# ---------------------------------------------------------------------------
# Part 4b - rolling-origin evaluation, model selection on earlier years,
#           MAE/RMSE trade-off, feature ablations, data-quality flags
# ---------------------------------------------------------------------------
JUMP_FLAG_YEARS = 10.0  # single-year change in e0 above this marks an unstable source series
FIRST_TARGET = 2019     # 2018 is the first year with three lags, so 2019 is the first origin with training data
ABLATIONS = {
    "Full": FORECAST_FEATURES,
    "No momentum (d1, d2, e0_d1)": [f for f in FORECAST_FEATURES if f not in ("d1", "d2", "e0_d1")],
    "No deep lags (ex_lag2, ex_lag3)": [f for f in FORECAST_FEATURES if f not in ("ex_lag2", "ex_lag3")],
    "No mortality (log_qx_lag1)": [f for f in FORECAST_FEATURES if f != "log_qx_lag1"],
    "No country e0 (e0_lag1, e0_d1)": [f for f in FORECAST_FEATURES if f not in ("e0_lag1", "e0_d1")],
    "Level only (age, ex_lag1)": ["age_start", "ex_lag1"],
}


def flag_unstable_series(lt):
    e0 = lt[lt["age_start"] == 0].pivot(index="country", columns="year", values="ex")
    jump = e0.diff(axis=1).abs().max(axis=1)
    flagged = jump[jump > JUMP_FLAG_YEARS].sort_values(ascending=False)
    pd.DataFrame({"max_abs_year_change_e0": jump.round(2), "flagged": jump > JUMP_FLAG_YEARS}) \
        .sort_values("max_abs_year_change_e0", ascending=False).to_csv(TAB / "data_quality_flags.csv")
    return flagged


def error_summary(t, models):
    rows = []
    for n in models:
        e = (t[n] - t["ex"]).abs()
        rows.append({"Model": n, "MAE": e.mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "median_AE": e.median(), "p95_AE": e.quantile(0.95), "max_AE": e.max()})
    return pd.DataFrame(rows)


def part4b_rolling(lt):
    print("\n=== Part 4b: rolling-origin forecasting evaluation ===")
    flagged = flag_unstable_series(lt)
    print(f"Unstable series (|year change in e0| > {JUMP_FLAG_YEARS:g} y):", flagged.round(1).to_dict())
    excl = list(flagged.index)
    panel = build_panel(lt)
    years = list(range(FIRST_TARGET, REF_YEAR + 1))

    # 1) every feasible origin, flagged series excluded from training and evaluation
    res, per = [], {}
    for y in years:
        t = forecast_year(panel, lt, y, exclude=excl)
        t = t[~t["country"].isin(excl)]
        per[y] = t
        s = error_summary(t, FORECAST_MODELS)
        s.insert(0, "target_year", y)
        res.append(s)
    roll = pd.concat(res, ignore_index=True)

    # 2) select the model on earlier origins only (by MAE and by RMSE), then score it on year y
    sel_rows = []
    for y in years[1:]:
        past = roll[roll["target_year"] < y].groupby("Model")[["MAE", "RMSE"]].mean()
        for crit in ("MAE", "RMSE"):
            chosen = past[crit].idxmin()
            now = roll[(roll["target_year"] == y) & (roll["Model"] == chosen)].iloc[0]
            sel_rows.append({"target_year": y, "criterion": crit, "selected": chosen,
                             "MAE": now["MAE"], "RMSE": now["RMSE"]})
    sel = pd.DataFrame(sel_rows)
    roll.round(3).to_csv(TAB / "forecast_rolling_origin.csv", index=False)
    sel.round(3).to_csv(TAB / "forecast_rolling_selection.csv", index=False)
    print(roll.pivot(index="Model", columns="target_year", values="MAE").round(2).to_string())
    print(roll.pivot(index="Model", columns="target_year", values="RMSE").round(2).to_string())
    print(sel.round(2).to_string(index=False))

    # per-country MAE averaged over all origins (retrospective, used in the appendix)
    allp = pd.concat(per.values())
    pc = allp.groupby("country").apply(
        lambda g: pd.Series({n: (g[n] - g["ex"]).abs().mean() for n in FORECAST_MODELS}),
        include_groups=False)
    pc["best_model"] = pc[FORECAST_MODELS].idxmin(axis=1)
    pc["best_MAE"] = pc[FORECAST_MODELS].min(axis=1)
    pc.round(3).to_csv(TAB / "forecast_rolling_by_country.csv")
    print("Best model per country (rolling):", pc["best_model"].value_counts().to_dict())

    # 3) pooled error distribution over all origins (MAE vs RMSE trade-off)
    pooled = error_summary(allp, FORECAST_MODELS)
    pooled.round(3).to_csv(TAB / "forecast_rolling_error_distribution.csv", index=False)
    print(pooled.round(2).to_string(index=False))

    # 4) feature ablations for the learned models over all origins
    ab_rows = []
    learned = {k: v for k, v in forecast_models().items() if k != "Ridge"}
    for label, feats in ABLATIONS.items():
        for y in years:
            t = forecast_year(panel, lt, y, features=feats, exclude=excl, models=learned)
            t = t[~t["country"].isin(excl)]
            for n in learned:
                e = (t[n] - t["ex"]).abs()
                ab_rows.append({"ablation": label, "Model": n, "target_year": y,
                                "MAE": e.mean(), "RMSE": float(np.sqrt((e ** 2).mean()))})
    ab = pd.DataFrame(ab_rows)
    ab.round(3).to_csv(TAB / "forecast_feature_ablation.csv", index=False)
    print(ab.pivot_table(index=["ablation", "Model"], columns="target_year", values="MAE").round(2).to_string())

    # 5) flagged series: what each model does on them (evaluation only)
    fl = []
    for y in years:
        t = forecast_year(panel, lt, y, exclude=excl)
        t = t[t["country"].isin(excl)]
        s = error_summary(t, FORECAST_MODELS)
        s.insert(0, "target_year", y)
        fl.append(s)
    fl = pd.concat(fl, ignore_index=True)
    fl.round(3).to_csv(TAB / "forecast_flagged_series_errors.csv", index=False)
    return {"flagged": flagged, "roll": roll, "sel": sel, "pooled": pooled, "ablation": ab,
            "by_country": pc}


FORECAST_MODELS = ["Baseline", "Linear trend", "Ridge", "Random Forest", "Gradient Boosting"]


def part4_forecasting(lt):
    print("\n=== Part 4: forecasting life expectancy for every country ===")
    panel = build_panel(lt)
    overall, per_country, preds = [], [], {}
    for year in HOLDOUT_YEARS:
        t = forecast_year(panel, lt, year)
        preds[year] = t
        for name in FORECAST_MODELS:
            r = {"holdout_year": year, "Model": name}
            r.update(metrics(t["ex"], t[name]))
            e0 = t[t["age_start"] == 0]
            r["e0_MAE"] = mean_absolute_error(e0["ex"], e0[name])
            overall.append(r)
        pc = t.groupby(["country", "iso3"]).apply(
            lambda g: pd.Series({f"MAE_{n}": mean_absolute_error(g["ex"], g[n]) for n in FORECAST_MODELS}),
            include_groups=False,
        ).reset_index()
        e0 = t[t["age_start"] == 0].set_index("country")
        for n in FORECAST_MODELS:
            pc[f"e0_abs_err_{n}"] = pc["country"].map((e0[n] - e0["ex"]).abs())
        mae_cols = [f"MAE_{n}" for n in FORECAST_MODELS]
        pc["best_model"] = pc[mae_cols].idxmin(axis=1).str.replace("MAE_", "", regex=False)
        pc.insert(0, "holdout_year", year)
        per_country.append(pc)
    overall = pd.DataFrame(overall)
    per_country = pd.concat(per_country, ignore_index=True)
    overall.round(3).to_csv(TAB / "forecast_overall_metrics.csv", index=False)
    per_country.round(3).to_csv(TAB / "forecast_metrics_by_country.csv", index=False)
    pd.concat(preds.values()).loc[:, ["country", "iso3", "year", "age_group", "ex"] + FORECAST_MODELS] \
        .round(3).to_csv(TAB / "forecast_predictions.csv", index=False)
    print(overall.round(3).to_string(index=False))
    for y in HOLDOUT_YEARS:
        print(f"Best model per country ({y}):",
              per_country[per_country["holdout_year"] == y]["best_model"].value_counts().to_dict())

    # 2024 projection with the best model on the latest hold-out year
    last = overall[overall["holdout_year"] == max(HOLDOUT_YEARS)].sort_values("MAE")
    best = last.iloc[0]["Model"]
    best_ml = last[last["Model"].isin(forecast_models())].iloc[0]["Model"]
    panel_f = build_panel(lt, extra_year=REF_YEAR + 1)
    train = panel_f[panel_f["ex_lag3"].notna() & panel_f["ex"].notna()]
    fut = panel_f[panel_f["year"] == REF_YEAR + 1].copy()
    m = clone(forecast_models()[best_ml]).fit(train[FORECAST_FEATURES], train["target_delta"])
    fut["ex_projection"] = fut["ex_lag1"] + m.predict(fut[FORECAST_FEATURES])
    fut["ex_2023"] = fut["ex_lag1"]
    fut[["country", "iso3", "age_group", "ex_2023", "ex_projection"]].round(3).to_csv(
        TAB / f"projection_{REF_YEAR + 1}_{safe_name(best_ml)}.csv", index=False)
    print(f"Best overall on {max(HOLDOUT_YEARS)}: {best}; projection {REF_YEAR + 1} uses {best_ml}")

    # Figure: overall comparison per hold-out year + per-country MAE distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, year in zip(axes[:2], HOLDOUT_YEARS):
        o = overall[overall["holdout_year"] == year].set_index("Model").loc[FORECAST_MODELS[::-1]]
        ax.barh(o.index, o["MAE"], color=[MODEL_COLORS[n] for n in o.index], height=0.6)
        for i, v in enumerate(o["MAE"]):
            ax.annotate(f"{v:.3f}", (v, i), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set(title=f"Forecast {year}: MAE over all countries x ages", xlabel="MAE (years)")
        ax.grid(axis="y", visible=False)
    ax = axes[2]
    pc = per_country[per_country["holdout_year"] == max(HOLDOUT_YEARS)]
    bp = ax.boxplot([pc[f"MAE_{n}"] for n in FORECAST_MODELS], orientation="horizontal", patch_artist=True,
                    widths=0.55, medianprops={"color": INK},
                    flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for patch, n in zip(bp["boxes"], FORECAST_MODELS):
        patch.set_facecolor(MODEL_COLORS[n])
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(FORECAST_MODELS) + 1), FORECAST_MODELS)
    ax.set_xscale("log")
    ax.set(title=f"Per-country MAE, {max(HOLDOUT_YEARS)} (log scale)", xlabel="MAE (years)")
    ax.grid(axis="y", visible=False)
    savefig(fig, FIG / "06_forecast_model_comparison.png")

    # Figure: Kyrgyzstan forecast vs actual by age
    fig, axes = plt.subplots(1, len(HOLDOUT_YEARS), figsize=(13, 4), sharey=True)
    for ax, year in zip(axes, HOLDOUT_YEARS):
        k = preds[year][preds[year]["country"] == REF_COUNTRY].sort_values("age_start")
        for n in FORECAST_MODELS:
            ax.plot(k["age_start"], k[n] - k["ex"], color=MODEL_COLORS[n], label=n, linewidth=1.6)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set(title=f"Kyrgyzstan {year}: forecast error by age", xlabel="Age (start of group)")
    axes[0].set_ylabel("Predicted - actual e(x) (years)")
    axes[-1].legend()
    savefig(fig, FIG / "07_kyrgyzstan_forecast_errors.png")
    return {"overall": overall, "per_country": per_country, "preds": preds,
            "projection": fut, "best": best, "best_ml": best_ml}


# ---------------------------------------------------------------------------
# Part 5 - country summary, per-country plots, comparison and insights
# ---------------------------------------------------------------------------
def country_summary(lt, cohort, fc):
    e0 = lt[lt["age_start"] == 0].pivot(index=["iso3", "country"], columns="year", values="ex")
    s = pd.DataFrame(index=e0.index)
    for y in (2015, 2019, 2021, REF_YEAR):
        s[f"e0_{y}"] = e0[y]
    covid_min = e0[[2020, 2021]].min(axis=1)
    s["covid_trough_year"] = e0[[2020, 2021]].idxmin(axis=1)
    s["covid_drop_2019_to_trough"] = e0[2019] - covid_min
    s["recovery_trough_to_2023"] = e0[REF_YEAR] - covid_min
    s["above_2019_level_in_2023"] = e0[REF_YEAR] >= e0[2019]
    s["change_2015_2023"] = e0[REF_YEAR] - e0[2015]
    s["rank_e0_2023"] = e0[REF_YEAR].rank(ascending=False, method="min").astype(int)
    ref = lt[lt["year"] == REF_YEAR]
    s["e65_2023"] = ref[ref["age_start"] == 65].set_index(["iso3", "country"])["ex"]
    lx = ref.pivot(index=["iso3", "country"], columns="age_start", values="lx")
    s["adult_mortality_45q15"] = 1 - lx[60] / lx[15]
    s = s.reset_index()
    k = s[s["country"] == REF_COUNTRY].iloc[0]
    s["gap_e0_vs_KGZ"] = s[f"e0_{REF_YEAR}"] - k[f"e0_{REF_YEAR}"]
    s["gap_e65_vs_KGZ"] = s["e65_2023"] - k["e65_2023"]

    pc = fc["per_country"]
    pc = pc[pc["holdout_year"] == max(HOLDOUT_YEARS)].drop(columns=["holdout_year", "iso3"])
    pc = pc.rename(columns={c: f"fc{max(HOLDOUT_YEARS)}_{c}" for c in pc.columns if c != "country"})
    proj = fc["projection"]
    proj = proj[proj["age_start"] == 0].set_index("country")["ex_projection"].rename(
        f"e0_{REF_YEAR + 1}_projection")
    s = (s.merge(cohort.drop(columns=["iso3"]), on="country", how="left")
          .merge(pc, on="country", how="left")
          .merge(proj, left_on="country", right_index=True, how="left"))
    s = s.sort_values(f"e0_{REF_YEAR}", ascending=False)
    # Kyrgyzstan first (starting point), then all countries by e0
    s = pd.concat([s[s["country"] == REF_COUNTRY], s[s["country"] != REF_COUNTRY]])
    s.round(3).to_csv(TAB / "country_summary.csv", index=False)
    return s


def plot_country(country, lt, summary, fc, kg_ref):
    c = lt[lt["country"] == country]
    r = summary[summary["country"] == country].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    fig.suptitle(
        f"{country} ({r['iso3']}) - e0 {REF_YEAR}: {r[f'e0_{REF_YEAR}']:.1f} y, rank "
        f"{r['rank_e0_2023']}/{len(summary)}, gap vs Kyrgyzstan {r['gap_e0_vs_KGZ']:+.1f} y, "
        f"45q15 {r['adult_mortality_45q15']:.3f}",
        fontsize=11, fontweight="bold", x=0.01, ha="left")

    ax = axes[0]
    for color, year in zip(YEAR_RAMP, [2015, 2019, 2021, 2023]):
        s = c[c["year"] == year]
        ax.plot(s["age_start"], s["ex"], color=color, label=str(year), linewidth=1.8)
    if country != REF_COUNTRY:
        ax.plot(kg_ref.index, kg_ref.values, color=REF_COLOR, linestyle="--", linewidth=1.6,
                label=f"Kyrgyzstan {REF_YEAR}")
    ax.set(title="Remaining life expectancy e(x)", xlabel="Age (start of group)", ylabel="Years")
    ax.legend()

    ax = axes[1]
    e0 = c[c["age_start"] == 0]
    kg0 = lt[(lt["country"] == REF_COUNTRY) & (lt["age_start"] == 0)]
    ax.plot(e0["year"], e0["ex"], color=COUNTRY_COLOR, marker="o", markersize=5, label=country)
    if country != REF_COUNTRY:
        ax.plot(kg0["year"], kg0["ex"], color=REF_COLOR, linestyle="--", linewidth=1.6,
                label="Kyrgyzstan")
    pj = r[f"e0_{REF_YEAR + 1}_projection"]
    ax.plot([REF_YEAR, REF_YEAR + 1], [r[f"e0_{REF_YEAR}"], pj], color=COUNTRY_COLOR,
            linestyle=":", marker="o", markersize=6, markerfacecolor=SURFACE,
            label=f"{REF_YEAR + 1} projection ({fc['best_ml']})")
    end_label(ax, REF_YEAR + 1, pj, f"{pj:.1f}")
    ax.set(title="Life expectancy at birth", xlabel="Year", ylabel="e0 (years)")
    ax.legend(loc="best")

    ax = axes[2]
    year = max(HOLDOUT_YEARS)
    vals = [r[f"fc{year}_MAE_{n}"] for n in FORECAST_MODELS]
    ax.barh(FORECAST_MODELS[::-1], vals[::-1], color=[MODEL_COLORS[n] for n in FORECAST_MODELS[::-1]],
            height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set(title=f"{year} forecast MAE by model (best: {r[f'fc{year}_best_model']})", xlabel="MAE (years)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(COUNTRY_FIG / f"{r['iso3']}_{safe_name(country)}.png", dpi=110)
    plt.close(fig)


def part5_comparison(lt, summary, fc):
    print("\n=== Part 5: country plots and cross-country comparison ===")
    kg_ref = life_table_for(lt, REF_COUNTRY)
    for i, country in enumerate(summary["country"], 1):
        plot_country(country, lt, summary, fc, kg_ref)
        if i % 40 == 0:
            print(f"  {i}/{len(summary)} country figures")

    k = summary[summary["country"] == REF_COUNTRY].iloc[0]

    # Distribution of e0 2023 with Kyrgyzstan and peers
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    ax = axes[0]
    ax.hist(summary[f"e0_{REF_YEAR}"], bins=30, color=MUTED, edgecolor=SURFACE, linewidth=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    top = ax.get_ylim()[1]
    ax.axvline(k[f"e0_{REF_YEAR}"], color=COUNTRY_COLOR, linewidth=2)
    ax.annotate(f"Kyrgyzstan {k[f'e0_{REF_YEAR}']:.1f}\nrank {k['rank_e0_2023']}/{len(summary)}",
                (k[f"e0_{REF_YEAR}"], top * 0.97), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK2, bbox=LABEL_BOX)
    med = summary[f"e0_{REF_YEAR}"].median()
    ax.axvline(med, color=INK2, linestyle="--", linewidth=1)
    ax.annotate(f"median {med:.1f}", (med, top * 0.97), xytext=(6, 0), textcoords="offset points",
                va="top", fontsize=8, color=INK2, bbox=LABEL_BOX)
    ax.set(title=f"Life expectancy at birth, {REF_YEAR} (192 countries)", xlabel="e0 (years)",
           ylabel="Countries")

    ax = axes[1]
    e0 = lt[lt["age_start"] == 0]
    world = e0.groupby("year")["ex"].median()
    ax.plot(world.index, world.values, color=MUTED, linestyle="--", linewidth=1.6, label="World median")
    labels, ends = ["World median"], [world.values[-1]]
    for color, country in zip(SERIES, [REF_COUNTRY] + PEERS):
        s = e0[e0["country"] == country]
        ax.plot(s["year"], s["ex"], color=color, linewidth=2.4 if country == REF_COUNTRY else 1.6,
                label=country)
        labels.append(country.replace(" Federation", ""))
        ends.append(s["ex"].iloc[-1])
    for text, y_end, y_lbl in zip(labels, ends, spread(ends, 0.28)):
        ax.annotate(text, (REF_YEAR, y_end), xytext=(REF_YEAR + 0.15, y_lbl), fontsize=8,
                    color=INK2, va="center")
    ax.set_xlim(2015, REF_YEAR + 1.6)
    ax.set(title="Kyrgyzstan vs Central Asian peers: e0", xlabel="Year", ylabel="e0 (years)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    savefig(fig, FIG / "08_kyrgyzstan_vs_world_and_peers.png")

    # COVID shock vs recovery
    fig, ax = plt.subplots(figsize=(7.5, 6))
    oth = summary[~summary["country"].isin(PEERS + [REF_COUNTRY])]
    ax.scatter(oth["covid_drop_2019_to_trough"], oth["recovery_trough_to_2023"], s=22, color=MUTED,
               alpha=0.6, edgecolor=SURFACE, linewidth=0.5, label="Other countries")
    pe = summary[summary["country"].isin(PEERS)]
    ax.scatter(pe["covid_drop_2019_to_trough"], pe["recovery_trough_to_2023"], s=55, color=REF_COLOR,
               edgecolor=SURFACE, linewidth=1.5, label="Central Asian peers")
    ax.scatter(k["covid_drop_2019_to_trough"], k["recovery_trough_to_2023"], s=80, color=COUNTRY_COLOR,
               edgecolor=SURFACE, linewidth=2, label="Kyrgyzstan", zorder=5)
    lab = pd.concat([pe, summary[summary["country"] == REF_COUNTRY]])
    lab_y = spread(lab["recovery_trough_to_2023"].to_numpy(), 0.35)
    for (_, r), y_lbl in zip(lab.iterrows(), lab_y):
        ax.annotate(r["country"].replace(" Federation", ""),
                    (r["covid_drop_2019_to_trough"], r["recovery_trough_to_2023"]),
                    xytext=(r["covid_drop_2019_to_trough"] - 0.25, y_lbl), ha="right", va="center",
                    fontsize=8, color=INK2, bbox=LABEL_BOX,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.8})
    # Clip extreme outliers (small or conflict-affected populations) and say so
    xlim, ylim = (-2, 7.5), (-3, 8.5)
    out = summary[~summary["covid_drop_2019_to_trough"].between(*xlim)
                  | ~summary["recovery_trough_to_2023"].between(*ylim)]
    ax.plot([0, 7.5], [0, 7.5], color=AXIS, linestyle="--", linewidth=1)
    ax.annotate("recovery = drop", (6.2, 6.2), xytext=(6, -2), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set(xlim=xlim, ylim=ylim,
           title="COVID-era e0 drop (2019 to 2020/21 trough) vs recovery by 2023",
           xlabel="Drop in e0 (years)", ylabel="Recovery in e0 (years)")
    if len(out):
        ax.annotate(f"Off-scale, not shown: {', '.join(out['country'])}", (0.01, 0.01),
                    xycoords="axes fraction", fontsize=7, color=MUTED)
    ax.legend(loc="upper left")
    savefig(fig, FIG / "09_covid_shock_vs_recovery.png")

    # Top / bottom / peers ranking
    s = summary.sort_values(f"e0_{REF_YEAR}", ascending=False)
    pick = pd.concat([s.head(10), s[s["country"].isin(PEERS + [REF_COUNTRY])], s.tail(10)])
    pick = pick.drop_duplicates("country").sort_values(f"e0_{REF_YEAR}")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [COUNTRY_COLOR if c == REF_COUNTRY else (REF_COLOR if c in PEERS else MUTED)
              for c in pick["country"]]
    ax.barh(pick["country"] + "  (#" + pick["rank_e0_2023"].astype(str) + ")", pick[f"e0_{REF_YEAR}"],
            color=colors, height=0.7)
    for i, v in enumerate(pick[f"e0_{REF_YEAR}"]):
        ax.annotate(f"{v:.1f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_xlim(pick[f"e0_{REF_YEAR}"].min() - 5, pick[f"e0_{REF_YEAR}"].max() + 3)
    ax.set(title=f"e0 {REF_YEAR}: top 10, bottom 10, Kyrgyzstan (blue) and peers (orange)",
           xlabel="e0 (years)")
    ax.grid(axis="y", visible=False)
    savefig(fig, FIG / "10_ranking_top_bottom_peers.png")


def part6_country_feature_importance(df, lt, strict):
    """Feature importance for every country (mean |SHAP| of Gradient Boosting).

    full   - participant GB on all 12 features, labels from that country's life table
    lf     - leakage-free GB (no age, no age proxies) + its CV gain over the baseline
    fc     - global forecasting GB (trained on years < last year), SHAP on that country's rows
    """
    print("\n=== Part 6: feature importance for every country ===")
    X_full, X_lf = df[FEATURES], df[strict]
    gb_full = participant_models()["Gradient Boosting"]
    lf_models = leakage_free_models()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    year = max(HOLDOUT_YEARS)
    panel = build_panel(lt)
    unstable = list(flag_unstable_series(lt).index)  # trained without unstable series, as in Part 4b
    train = panel[(panel["year"] < year) & panel["ex_lag3"].notna() & panel["ex"].notna()
                  & ~panel["country"].isin(unstable)]
    test = panel[panel["year"] == year]
    fc_model = clone(forecast_models()["Gradient Boosting"]).fit(train[FORECAST_FEATURES],
                                                                train["target_delta"])
    fc_sv = pd.DataFrame(np.abs(shap.TreeExplainer(fc_model).shap_values(test[FORECAST_FEATURES])),
                         columns=FORECAST_FEATURES, index=test.index)
    fc_imp = fc_sv.groupby(test["country"]).mean()

    rows = []
    for i, country in enumerate(sorted(lt["country"].unique()), 1):
        yc = base_le(df["age"], life_table_for(lt, country))
        full = clone(gb_full).fit(X_full, yc)
        s_full = pd.Series(np.abs(shap.TreeExplainer(full).shap_values(X_full)).mean(axis=0), index=FEATURES)
        non_age = s_full.drop("age")

        mae = {n: mean_absolute_error(yc, cross_val_predict(clone(m), X_lf, yc, cv=kf))
               for n, m in lf_models.items() if n != "Random Forest"}
        lf = clone(lf_models["Gradient Boosting"]).fit(X_lf, yc)
        s_lf = pd.Series(np.abs(shap.TreeExplainer(lf).shap_values(X_lf)).mean(axis=0),
                         index=strict).sort_values(ascending=False)

        f = fc_imp.loc[country].sort_values(ascending=False)
        rows.append({
            "country": country,
            "full_age_shap_share_%": 100 * s_full["age"] / s_full.sum(),
            "full_top_non_age_feature": non_age.idxmax(),
            "full_top_non_age_share_%": 100 * non_age.max() / s_full.sum(),
            "lf_cv_MAE_GB": mae["Gradient Boosting"],
            "lf_cv_MAE_Baseline": mae["Baseline"],
            "lf_gain_vs_baseline_%": 100 * (1 - mae["Gradient Boosting"] / mae["Baseline"]),
            "lf_top1": s_lf.index[0],
            "lf_top2": s_lf.index[1],
            "fc_top1": f.index[0],
            "fc_top1_share_%": 100 * f.iloc[0] / f.sum(),
            "fc_top2": f.index[1],
        })
        if i % 40 == 0:
            print(f"  {i} countries")
    fi = pd.DataFrame(rows)
    fi.round(3).to_csv(TAB / "country_feature_importance.csv", index=False)
    print("Leakage-free GB beats baseline in", int((fi["lf_gain_vs_baseline_%"] > 0).sum()), "countries;",
          "forecast top feature:", fi["fc_top1"].value_counts().to_dict())
    return fi


def write_insights(summary, p2, p2b, cohort, fc):
    k = summary[summary["country"] == REF_COUNTRY].iloc[0]
    n = len(summary)
    res = p2["results"].set_index("Model")
    abl = p2["ablation"]
    o = fc["overall"]
    y = max(HOLDOUT_YEARS)
    oy = o[o["holdout_year"] == y].set_index("Model")
    bm = fc["per_country"][fc["per_country"]["holdout_year"] == y]["best_model"].value_counts()
    peers = summary[summary["country"].isin(PEERS)].set_index("country")
    rec = summary["above_2019_level_in_2023"].mean()

    lines = [
        "# Life expectancy analysis - Kyrgyzstan as starting point",
        "",
        "## Kyrgyzstan",
        f"- e0 {REF_YEAR}: {k[f'e0_{REF_YEAR}']:.2f} years, rank {k['rank_e0_2023']}/{n}; "
        f"change 2015-2023: {k['change_2015_2023']:+.2f}; e65: {k['e65_2023']:.2f}; "
        f"45q15: {k['adult_mortality_45q15']:.3f}.",
        f"- COVID trough {int(k['covid_trough_year'])}: drop {k['covid_drop_2019_to_trough']:.2f} y, "
        f"recovery {k['recovery_trough_to_2023']:.2f} y (above 2019 level: {bool(k['above_2019_level_in_2023'])}).",
        f"- World median e0 {REF_YEAR}: {summary[f'e0_{REF_YEAR}'].median():.2f}.",
        "- Peers (e0 2023, gap vs KGZ): " + "; ".join(
            f"{c} {r[f'e0_{REF_YEAR}']:.1f} ({r['gap_e0_vs_KGZ']:+.1f})" for c, r in peers.iterrows()),
        "",
        "## Participant models (Kyrgyzstan 2023 life table as target)",
    ]
    for m in res.index:
        lines.append(
            f"- {m}: hold-out MAE {res.loc[m, 'MAE_real']:.3f} (aug {res.loc[m, 'MAE_aug']:.3f}), "
            f"CV MAE {res.loc[m, 'CV_MAE']:.3f} +/- {res.loc[m, 'CV_MAE_sd']:.3f}, CV R2 {res.loc[m, 'CV_R2']:.3f}")
    for fs in abl["Feature set"].unique():
        a = abl[abl["Feature set"] == fs].set_index("Model")
        lines.append(f"- {fs}: " + ", ".join(f"{m} MAE {a.loc[m, 'CV_MAE']:.2f}" for m in a.index))
    sh = p2["shap"]
    lines += [
        f"- Top RF features: " + ", ".join(p2["importance"]["feature"].head(4)),
        f"- SHAP (leaky, all features): age share RF {sh.loc['age', 'share_RF_%']:.0f}%, "
        f"GB {sh.loc['age', 'share_GB_%']:.0f}% of total mean |SHAP|.",
        "",
        "## Leakage-free RF and GB (Part 2b)",
        f"- Age proxies (|rho| >= {AGE_PROXY_THRESHOLD}): {', '.join(p2b['proxies'])}; "
        f"strict feature set: {', '.join(p2b['strict'])}",
    ]
    lf = p2b["results"]
    for sc in lf["Scenario"].unique():
        r = lf[lf["Scenario"] == sc].set_index("Model")
        parts = []
        for m in r.index:
            s = f"{m} MAE {r.loc[m, 'MAE']:.2f}"
            if m != "Baseline":
                s += f" ({r.loc[m, 'MAE_gain_vs_baseline_%']:+.0f}% vs baseline"
                s += "" if pd.isna(r.loc[m, "permutation_p_value"]) else f", p={r.loc[m, 'permutation_p_value']:.3f}"
                s += ")"
            parts.append(s)
        lines.append(f"- {sc}: " + "; ".join(parts))
    for key, lbl in [("shap_no_age", "No age"), ("shap_strict", "No age, no proxies")]:
        s = p2b[key]
        lines.append(f"- SHAP top features ({lbl}): RF " + ", ".join(
            s["Random Forest"].sort_values(ascending=False).head(3).index) + "; GB " + ", ".join(
            s["Gradient Boosting"].sort_values(ascending=False).head(3).index))
    lines += [
        "",
        "## Same cohort, every country's life table",
        f"- Cohort mean remaining LE ranges {cohort['cohort_mean_remaining_LE'].min():.1f} - "
        f"{cohort['cohort_mean_remaining_LE'].max():.1f} y (KGZ "
        f"{cohort.set_index('country').loc[REF_COUNTRY, 'cohort_mean_remaining_LE']:.1f}).",
        f"- Best ML model by country: {cohort['cohort_best_model'].value_counts().to_dict()}",
        "",
        f"## Forecasting {y} for all countries",
    ]
    for m in FORECAST_MODELS:
        lines.append(f"- {m}: MAE {oy.loc[m, 'MAE']:.3f}, RMSE {oy.loc[m, 'RMSE']:.3f}, e0 MAE {oy.loc[m, 'e0_MAE']:.3f}")
    lines += [
        f"- Best model per country: {bm.to_dict()}",
        f"- Countries at/above their 2019 e0 by 2023: {rec:.0%}",
        f"- {REF_YEAR + 1} projection for Kyrgyzstan ({fc['best_ml']}): "
        f"{k[f'e0_{REF_YEAR + 1}_projection']:.2f}",
    ]
    (OUT / "insights.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    lt = load_life_table()
    df = load_participants()
    print(f"Life table: {lt['country'].nunique()} countries, {lt['year'].min()}-{lt['year'].max()}; "
          f"participants: {df.shape}")
    part1_kyrgyzstan(lt)
    p2 = part2_participants(df, lt)
    p2b = part2b_leakage_free(df, lt)
    part2c_review(df, lt)
    cohort = part3_cohort_by_country(df, lt)
    fc = part4_forecasting(lt)
    part4b_rolling(lt)
    summary = country_summary(lt, cohort, fc)
    part5_comparison(lt, summary, fc)
    part6_country_feature_importance(df, lt, p2b["strict"])
    write_insights(summary, p2, p2b, cohort, fc)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
