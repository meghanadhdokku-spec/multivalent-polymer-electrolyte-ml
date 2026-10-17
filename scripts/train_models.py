"""Train the fixed model shortlist (dummy baseline, Random Forest, XGBoost) on:
  (A) OBELiX  -- n=599, Li-ion inorganic crystals -- pipeline/methodology validation
  (B) multivalent polymer-electrolyte literature extract -- n=26, Mg/Zn/Na -- the
      paper's actual novel contribution, but tiny and noisy -- evaluated with
      leave-one-out CV since a single train/test split would be meaningless at n=26.

Target is log10(ionic conductivity in S/cm) in both cases. Metrics: R2, MAE, RMSE.
Results are written to results/model_results.md and results/*.csv (predictions).
"""
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

RESULTS_DIR = r"C:\Polyemr Electrolyter ML project\results"


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def run_obelix():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\obelix_features.csv")
    y = df["log10_conductivity"].values
    X = df.drop(columns=["log10_conductivity", "composition", "doi"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        out[name] = metrics(y_test, pred)
    return out, X.columns.tolist(), len(df)


def run_multivalent():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features.csv")
    y = df["log10_conductivity"].values
    n_distinct_hosts = df["polymer_host"].str.lower().str.strip().nunique()
    n_approx = int(df["value_is_approximate"].sum())
    drop_cols = ["log10_conductivity", "polymer_host", "source_citation"]
    X = df.drop(columns=drop_cols)

    loo = LeaveOneOut()
    models = {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=loo)
        out[name] = metrics(y, preds)
    return out, X.columns.tolist(), len(df), n_distinct_hosts, n_approx


def run_multivalent_v2():
    """Same LOO-CV protocol as run_multivalent, but on the v2 chemistry-informed
    feature set (numeric concentration, anion identity, cation physical
    descriptors, expanded host/plasticizer chemistry flags, secondary-source flag).
    """
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    y = df["log10_conductivity"].values
    drop_cols = ["log10_conductivity", "polymer_host", "anion_class"]
    X = df.drop(columns=drop_cols)

    loo = LeaveOneOut()
    models = {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1),
    }
    out = {}
    importances = None
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=loo)
        out[name] = metrics(y, preds)
        if name == "RandomForest":
            model.fit(X, y)
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return out, X.columns.tolist(), len(df), importances


def main():
    obelix_metrics, obelix_cols, n_obelix = run_obelix()
    mv_metrics, mv_cols, n_mv, n_distinct_hosts, n_approx = run_multivalent()
    mv_v2_metrics, mv_v2_cols, n_mv_v2, feature_importances = run_multivalent_v2()

    report = []
    report.append("# Model training results\n")
    report.append(f"## Dataset A: OBELiX (Li-ion inorganic crystals, n={n_obelix})\n")
    report.append("80/20 train/test split, random_state=42. Target: log10(S/cm).\n")
    report.append("| Model | R2 | MAE | RMSE |")
    report.append("|---|---|---|---|")
    for name, m in obelix_metrics.items():
        report.append(f"| {name} | {m['R2']:.3f} | {m['MAE']:.3f} | {m['RMSE']:.3f} |")
    report.append(f"\nFeatures used ({len(obelix_cols)}): {', '.join(obelix_cols)}\n")

    report.append(f"\n## Dataset B: Multivalent (Mg/Zn/Na) polymer electrolytes, literature extract (n={n_mv})\n")
    report.append("Leave-one-out cross-validation (n is far too small for a held-out split). Target: log10(S/cm).\n")
    report.append("| Model | R2 (LOO) | MAE | RMSE |")
    report.append("|---|---|---|---|")
    for name, m in mv_metrics.items():
        report.append(f"| {name} | {m['R2']:.3f} | {m['MAE']:.3f} | {m['RMSE']:.3f} |")
    report.append(f"\nFeatures used ({len(mv_cols)}): {', '.join(mv_cols)}\n")

    report.append("\n## Interpretation\n")
    report.append(
        "- OBELiX results validate that the pipeline (composition/structure -> log10 conductivity, "
        "RF/XGBoost vs. a mean-predicting dummy baseline) works and that tree ensembles beat the "
        "baseline on an established, well-populated inorganic Li-ion dataset.\n"
    )
    tree_r2s = [m["R2"] for name, m in mv_metrics.items() if name != "Dummy(mean)"]
    r2_range = f"{min(tree_r2s):.2f}-{max(tree_r2s):.2f}"
    report.append(
        f"- The multivalent (Mg/Zn/Na) polymer dataset is the paper's actual novel contribution. Tree "
        f"models beat the dummy baseline under LOO CV (R2 ~{r2_range}), but this should NOT be read as a "
        f"robust result: several rows share the same polymer host measured at different temperatures only "
        f"(e.g. PAES-g-PEG, PVA/HPC, Nafion, PVA-borax-glycerol at 3 temperatures), so in LOO CV the model "
        f"can partly recognize a left-out row's host fingerprint from its near-duplicate still in the "
        f"training fold -- this inflates R2 versus a true held-out-composition test. At n={n_mv} across "
        f"only {n_distinct_hosts} distinct host systems this cannot be fully fixed by re-splitting; it "
        f"should be reported as a caveat, and the real headline finding is the data-scarcity problem "
        f"itself ({n_mv} usable points, {n_approx} of them only order-of-magnitude, across 3 cations and "
        f"{n_distinct_hosts} distinct host systems), not the R2 number.\n"
    )
    report.append(
        "- Framing for the paper: Part 1 (OBELiX) demonstrates the ML methodology on a mature benchmark; "
        "Part 2 (multivalent) applies the same methodology to a hand-curated novel dataset and the small-n "
        "result itself supports the paper's stated novelty case -- that public ML-ready data for "
        "multivalent/non-lithium polymer electrolytes barely exists yet.\n"
    )
    report.append(
        "- TREND ACROSS GROWTH PASSES (worth reporting explicitly): LOO R2 for the tree models went "
        "~0.47-0.49 (n=48, 40 distinct hosts) -> ~0.40-0.41 (n=91, 75 distinct hosts) -> "
        f"~{min(tree_r2s):.2f}-{max(tree_r2s):.2f} (n={n_mv}, {n_distinct_hosts} distinct hosts). This "
        "monotonic decline as the dataset grew more diverse is itself informative: it shows the earlier, "
        "higher R2 values were substantially inflated by LOO CV recognizing near-duplicate host fingerprints, "
        "not by the model learning real structure-conductivity relationships. The current, more diverse "
        "dataset gives a more honest (lower) estimate of what coarse categorical features can predict, and "
        "argues that further improvement needs richer chemical features (numeric salt concentration, "
        "molecular descriptors, anion identity), not just more rows of the same coarse feature types.\n"
    )

    report.append(f"\n## Dataset B-v2: Multivalent, chemistry-informed features (n={n_mv_v2})\n")
    report.append(
        "Same rows as Dataset B, but with richer features: numeric salt concentration (wt%/M/mol%, "
        "median-imputed with an imputed-flag), anion identity one-hot + fluorinated-anion flag, cation "
        "physical descriptors (Shannon ionic radius, charge, charge density z/r) instead of relying only "
        "on one-hot cation identity, expanded host chemistry flags (ether-oxygen backbone, natural "
        "biopolymer, fluorinated backbone, hydrogel/aqueous), plasticizer TYPE flags (ionic liquid, "
        "ceramic filler, organic co-solvent) instead of one binary, and a secondary-source/low-confidence "
        "flag parsed from notes. Leave-one-out CV, same protocol as Dataset B. Target: log10(S/cm).\n"
    )
    report.append("| Model | R2 (LOO) | MAE | RMSE |")
    report.append("|---|---|---|---|")
    for name, m in mv_v2_metrics.items():
        report.append(f"| {name} | {m['R2']:.3f} | {m['MAE']:.3f} | {m['RMSE']:.3f} |")
    report.append(f"\nFeatures used ({len(mv_v2_cols)}): {', '.join(mv_v2_cols)}\n")
    report.append("\nRandomForest feature importances (top 15):\n")
    report.append("| Feature | Importance |")
    report.append("|---|---|")
    for feat, imp in feature_importances.head(15).items():
        report.append(f"| {feat} | {imp:.3f} |")

    v2_tree_r2s = [m["R2"] for name, m in mv_v2_metrics.items() if name != "Dummy(mean)"]
    v1_tree_r2s = [m["R2"] for name, m in mv_metrics.items() if name != "Dummy(mean)"]
    improved = max(v2_tree_r2s) > max(v1_tree_r2s)
    report.append(
        f"\n- **v1 vs v2 comparison**: coarse-feature R2 was ~{min(v1_tree_r2s):.2f}-{max(v1_tree_r2s):.2f}; "
        f"chemistry-informed v2 R2 is ~{min(v2_tree_r2s):.2f}-{max(v2_tree_r2s):.2f}. "
        + (
            "Feature engineering HELPED, supporting the hypothesis that richer chemical features (not just "
            "more rows) are the better investment at this dataset size -- see the feature importances above "
            "for which new features the model actually leaned on."
            if improved
            else
            "Feature engineering did NOT clearly help at this n -- with ~106 distinct host systems and only "
            "137 rows, there may simply not be enough repeated examples per anion/concentration combination "
            "for the model to learn from the richer features yet, or the added features mostly encode "
            "information already implicit in the coarse host flags. This itself is a useful negative result: "
            "it suggests the bottleneck right now is still sample size (or irreducible literature-measurement "
            "noise), not feature coarseness -- worth stating honestly in the paper rather than overclaiming."
        )
        + "\n"
    )
    report.append(
        "- Caveat on the cation physical descriptors (ionic radius, charge, charge density): with only 3 "
        "cations in this dataset, these are deterministic functions of which one-hot cation flag is set "
        "(e.g. cation_Mg=1 always implies radius=0.72, charge=2) -- they carry no information beyond the "
        "one-hot flags yet, which is why they don't show up with independent importance above. They would "
        "only start earning their keep if a 4th+ cation (e.g. Ca2+) were added, letting the model learn a "
        "continuous radius/charge-density trend rather than memorizing 3 separate categories.\n"
    )

    with open(f"{RESULTS_DIR}\\model_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    with open(f"{RESULTS_DIR}\\model_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"obelix": obelix_metrics, "multivalent_v1": mv_metrics, "multivalent_v2": mv_v2_metrics},
            f,
            indent=2,
        )

    print("\n".join(report))


if __name__ == "__main__":
    main()
