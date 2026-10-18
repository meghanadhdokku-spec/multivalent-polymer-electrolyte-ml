"""Final model pipeline for the multivalent (Mg/Zn/Na) polymer electrolyte
dataset. Supersedes model_results.md / tuning_results.md / nested_cv_results.md,
which remain only as the record of how the final hyperparameters were chosen.

Produces two things:
  1. results/final_model.pkl -- the trained RandomForest, fit on all 137 rows
     with the nested-CV-validated hyperparameters, ready to load and predict.
  2. results/FINAL_MODEL_REPORT.md -- final performance with caveats, what was
     tried, known limitations, and what would help most going forward.

Run order: build_obelix_dataset.py -> build_multivalent_dataset_v2.py -> this
script. tune_models.py and nested_cv.py were exploratory steps whose
conclusions are folded in here and don't need to be re-run.
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RESULTS_DIR = r"C:\Polyemr Electrolyter ML project\results"

# Chosen from tune_models.py's 5-fold grid search, confirmed by nested_cv.py
# to NOT be an artifact of hyperparameter-selection leakage (see nested_cv_results.md).
BEST_PARAMS = {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def main():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    y = df["log10_conductivity"].values
    feature_cols = [c for c in df.columns if c not in ("log10_conductivity", "polymer_host", "anion_class")]
    X = df[feature_cols]

    n_distinct_hosts = df["polymer_host"].str.lower().str.strip().nunique()
    n_conc_known = int(df["concentration_known"].sum())

    # final reported performance: plain LOO with the fixed, already-chosen
    # hyperparameters (this is NOT re-tuned per fold -- that nested estimate
    # was already computed in nested_cv.py and confirmed ~equal to this).
    loo = LeaveOneOut()
    dummy_preds = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=loo)
    rf_preds = cross_val_predict(RandomForestRegressor(**BEST_PARAMS), X, y, cv=loo)
    dummy_metrics = metrics(y, dummy_preds)
    rf_metrics = metrics(y, rf_preds)

    # fit the final deployable model on ALL available data
    final_model = RandomForestRegressor(**BEST_PARAMS)
    final_model.fit(X, y)
    importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    with open(f"{RESULTS_DIR}\\final_model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "feature_columns": feature_cols}, f)

    with open(f"{RESULTS_DIR}\\model_results.json") as f:
        obelix_metrics = json.load(f)["obelix"]

    report = []
    report.append("# FINAL MODEL REPORT -- multivalent (Mg/Zn/Na) polymer electrolyte conductivity\n")
    report.append(
        "This supersedes model_results.md / tuning_results.md / nested_cv_results.md as the single "
        "reference. Those files remain as the record of how this result was reached (dataset growth -> "
        "feature engineering -> hyperparameter tuning -> leakage check).\n"
    )

    report.append("## Bottom line\n")
    report.append(
        f"- **Final model: RandomForestRegressor({BEST_PARAMS}), trained on {len(df)} literature-mined "
        f"Mg/Zn/Na polymer-electrolyte data points.**\n"
        f"- **Validated performance (leave-one-out CV): R2 = {rf_metrics['R2']:.3f}**, vs a mean-predicting "
        f"dummy baseline R2 = {dummy_metrics['R2']:.3f}. MAE = {rf_metrics['MAE']:.3f} log10(S/cm) "
        f"(i.e. typical prediction is within roughly a factor of {10**rf_metrics['MAE']:.1f}x of the true "
        f"conductivity).\n"
        f"- **Caveat carried forward honestly**: a stricter 5-fold CV estimate (computed during "
        f"hyperparameter selection, see tuning_results.md) gave R2 = 0.376 -- notably lower. The gap is "
        f"best explained by 5-fold's higher variance at n={len(df)} (holding out ~27 points at once vs 1 "
        f"for LOO), not by leakage (nested_cv.py's fully unbiased check gave R2=0.478, matching this "
        f"result, not the 5-fold one). **Report both numbers in the paper**: ~0.49 as the headline, ~0.38 "
        f"as a deliberately conservative lower bound.\n"
    )
    report.append(
        f"- For context, the SAME pipeline on OBELiX (n=599, Li-ion inorganic crystals, a mature benchmark "
        f"with clean/consistent measurements) reaches R2={obelix_metrics['XGBoost']['R2']:.2f}. The gap "
        f"between that and ~0.49 here is the honest measure of how much harder -- and how much more "
        f"data-starved -- the multivalent/polymer problem is, which is the paper's actual point.\n"
    )

    report.append("\n## What the model is built on\n")
    report.append(
        f"- {len(df)} rows across 3 cations (Mg, Zn, Na) and {n_distinct_hosts} distinct polymer-host "
        "systems, hand-curated from the literature across several rounds of targeted search and dataset "
        "growth. 2 alkaline/KOH-soak rows were excluded (they measure OH- transport, not the target "
        "cation).\n"
    )
    report.append(
        f"- Features (chemistry-informed, not just coarse categorical flags): numeric salt concentration "
        f"where parseable ({n_conc_known}/{len(df)} rows), anion identity (one-hot, rare categories with <=3 members pooled "
        "into 'rare_other') + a fluorinated-anion flag, cation physical descriptors (ionic radius, charge, "
        "charge density -- currently redundant with the one-hot cation flags at only 3 cations, would "
        "matter more with a 4th added later), expanded polymer-host chemistry flags (ether-oxygen "
        "backbone, natural biopolymer, fluorinated backbone, hydrogel/aqueous), plasticizer-type flags "
        "(ionic liquid / ceramic filler / organic co-solvent), and a secondary-source/low-confidence flag.\n"
    )
    report.append("\n## Feature importances (top 15)\n")
    report.append("| Feature | Importance |")
    report.append("|---|---|")
    for feat, imp in importances.head(15).items():
        report.append(f"| {feat} | {imp:.3f} |")

    report.append("\n## Known limitations\n")
    report.append(
        f"- **Sample size**: {len(df)} rows across {n_distinct_hosts}-ish distinct host systems is small for 46 features; most "
        "host systems appear only once or twice, so the model leans heavily on cation identity and a few "
        "coarse host-chemistry flags rather than learning fine structure-property trends.\n"
        "- **Measurement heterogeneity**: different papers, different temperature conventions (many "
        "'assumed RT'), different confidence levels (several rows are secondary-sourced from review tables, "
        "not independently verified against primary full text) -- this is irreducible noise that caps "
        "achievable R2 regardless of model or feature choices.\n"
        "- **LOO vs 5-fold discrepancy**: see Bottom Line above -- report both numbers, don't cherry-pick.\n"
        f"- **Concentration coverage**: only {n_conc_known}/{len(df)} rows have a parseable numeric salt concentration; the "
        "rest are median-imputed with a flag. More systematic concentration extraction would likely help "
        "more than further hyperparameter tuning at this point.\n"
        "- **Growth has diminishing returns**: several rounds of literature search have mined essentially "
        "every efficient review-table source found; further rows would mostly come from individual-paper "
        "search (expensive per row) rather than the efficient review-table approach that drove most of "
        "this growth.\n"
    )

    report.append("\n## Not yet done\n")
    report.append(
        "- No figures/plots, no manuscript drafting -- this stage covered dataset curation and modeling "
        "only.\n"
    )

    out_text = "\n".join(report)
    with open(f"{RESULTS_DIR}\\FINAL_MODEL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(out_text)
    print(out_text)
    print(f"\nSaved model artifact -> {RESULTS_DIR}\\final_model.pkl")


if __name__ == "__main__":
    main()
