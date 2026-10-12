# -*- coding: utf-8 -*-
"""Build LaTeX assets for paper/main.tex from the outputs of mlforle12.py.

Writes paper/appendix_countries.tex (prediction status + feature importance for
every country) and copies the figures used in the paper to paper/figures/.

Run after mlforle12.py:
  uv run --python 3.12 --with pandas python make_paper_assets.py
"""

import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
TAB = BASE / "outputs" / "tables"
FIG = BASE / "outputs" / "figures"
PAPER = BASE / "paper"
PAPER_FIG = PAPER / "life_figures"

FIGURES = []  # the paper's only figure (the framework diagram) is drawn in TikZ

MODEL_ABBR = {"Baseline": "Naive", "Linear trend": "Trend", "Ridge": "Ridge",
              "Random Forest": "RF", "Gradient Boosting": "GB"}
FEATURE_ABBR = {
    "age": "age", "sex": "sex", "BMI": "BMI", "sleep_hours": "sleep",
    "avg_steps_per_day": "steps", "physical_activity": "activity", "smoking_status": "smoking",
    "alcohol_units_per_week": "alcohol", "body_water_kg": "water",
    "muscle_mass_kg": "muscle", "body_fat_percent": "body fat", "chronic_disease": "chronic",
    # plain-text notation keeps appendix rows compact (no sub/superscripts)
    "age_start": "age grp.", "ex_lag1": "e(t-1)", "ex_lag2": "e(t-2)", "ex_lag3": "e(t-3)",
    "d1": "$\\Delta$(t-1)", "d2": "$\\Delta$(t-2)", "log_qx_lag1": "log q(t-1)",
    "e0_lag1": "e0(t-1)", "e0_d1": "$\\Delta$e0(t-1)",
}
SHORT_NAMES = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Dem. People's Republic of Korea": "DPR Korea",
    "Democratic Republic of the Congo": "DR Congo",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Lao PDR",
    "Saint Vincent and the Grenadines": "St Vincent \\& Gren.",
    "Saint Kitts and Nevis": "St Kitts \\& Nevis",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "USA",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Central African Republic": "Central African Rep.",
    "Sao Tome and Principe": "Sao Tome \\& Pr.",
    "Antigua and Barbuda": "Antigua \\& Barbuda",
    "Bosnia and Herzegovina": "Bosnia \\& Herz.",
    "Trinidad and Tobago": "Trinidad \\& Tobago",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "Republic of Moldova": "Moldova",
    "Republic of Korea": "Rep. of Korea",
    "United Arab Emirates": "UAE",
    "State of Palestine": "Palestine",
    "Brunei Darussalam": "Brunei",
    "Micronesia (Federated States of)": "Micronesia",
}


def forecastability(mae):
    return "High" if mae < 0.5 else ("Med." if mae < 1.5 else "Low")


def longtable(caption, label, colspec, header, rows, ncols, sep="1.6pt"):
    head = " & ".join(header) + " \\\\"
    out = [
        f"{{\\tiny\\setlength{{\\tabcolsep}}{{{sep}}}\\renewcommand{{\\arraystretch}}{{0.85}}",
        f"\\begin{{longtable}}{{{colspec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead",
        f"\\multicolumn{{{ncols}}}{{l}}{{\\emph{{(continued)}}}}\\\\",
        "\\toprule", head, "\\midrule", "\\endhead",
        "\\bottomrule", "\\endfoot",
    ]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\end{longtable}}"]
    return "\n".join(out) + "\n"


def main():
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for f in PAPER_FIG.glob("*.png"):  # drop figures no longer used by the paper
        if f.name not in FIGURES:
            f.unlink()
    for f in FIGURES:
        shutil.copy(FIG / f, PAPER_FIG / f)

    s = pd.read_csv(TAB / "country_summary.csv")
    fi = pd.read_csv(TAB / "country_feature_importance.csv").set_index("country")
    roll = pd.read_csv(TAB / "forecast_rolling_by_country.csv").set_index("country")  # 2019-2023 origins
    flags = pd.read_csv(TAB / "data_quality_flags.csv").set_index("country")["flagged"]

    rows = []
    for _, r in s.iterrows():
        f = fi.loc[r["country"]]
        c = SHORT_NAMES.get(r["country"], r["country"])
        if r["country"] == "Kyrgyzstan":
            c = "\\textbf{Kyrgyzstan}"
        if flags.get(r["country"], False):  # unstable source series: not evaluated
            c += "$^\\dagger$"
            status = ["--", "--", "--", "Unstable"]
        else:
            g = roll.loc[r["country"]]
            status = [MODEL_ABBR[g["best_model"]], f"{g['best_MAE']:.2f}", f"{g['Baseline']:.2f}",
                      forecastability(g["best_MAE"])]
        rows.append([
            c, f"{r['e0_2023']:.1f}", str(int(r["rank_e0_2023"])), f"{r['gap_e0_vs_KGZ']:+.1f}",
            "yes" if r["above_2019_level_in_2023"] else "no", *status,
            f"{f['full_age_shap_share_%']:.1f}", FEATURE_ABBR[f["full_top_non_age_feature"]],
            f"{f['lf_gain_vs_baseline_%']:+.0f}",
            f"{FEATURE_ABBR[f['lf_top1']]}, {FEATURE_ABBR[f['lf_top2']]}",
            f"{FEATURE_ABBR[f['fc_top1']]} ({f['fc_top1_share_%']:.0f})",
        ])
    header = ["Country", "$e^0_{23}$", "Rank", "$\\Delta$KGZ", "$\\geq$2019", "Best", "MAE",
              "Naive", "Band", "Age\\%", "Non-age top", "LF gain", "LF top-2",
              "Fcst. top (\\%)"]
    caption = "Prediction status and feature importance for all 192 countries (columns explained above)."
    (PAPER / "appendix_countries.tex").write_text(
        longtable(caption, "tab:app-countries", "l r r r c l r r l r l r l l", header, rows, 14,
                  sep="4pt"),
        encoding="utf-8")
    for old in ("appendix_status.tex", "appendix_importance.tex"):
        (PAPER / old).unlink(missing_ok=True)
    print("Wrote", PAPER / "appendix_countries.tex")


if __name__ == "__main__":
    main()
