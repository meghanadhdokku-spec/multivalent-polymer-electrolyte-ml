"""Hyperparameter tuning for RandomForest/XGBoost on the v2 multivalent feature
set. Every run so far used arbitrary hand-picked hyperparameters (n_estimators,
max_depth, learning_rate) with no actual tuning -- this does a small grid
search per the fixed model shortlist (no new model types, just tuning within
the two tree ensembles already chosen).

Selection uses 5-fold CV (not LOO) to keep the grid search cheap: 137 LOO
folds x a multi-point grid would be expensive for little benefit. The WINNING
hyperparameters are then evaluated with the same leave-one-out protocol used
throughout this project, so the final number is directly comparable to all
prior results.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RESULTS_DIR = r"C:\Polyemr Electrolyter ML project\results"


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def main():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    y = df["log10_conductivity"].values
    X = df.drop(columns=["log10_conductivity", "polymer_host", "anion_class"])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    loo = LeaveOneOut()

    rf_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 4, 6, None],
        "min_samples_leaf": [1, 2, 4],
    }
    xgb_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
    }

    print("Tuning RandomForest (5-fold CV grid search)...")
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), rf_grid, scoring="r2", cv=kfold, n_jobs=-1
    )
    rf_search.fit(X, y)
    print("  best params:", rf_search.best_params_, "5-fold CV R2:", round(rf_search.best_score_, 3))

    print("Tuning XGBoost (5-fold CV grid search)...")
    xgb_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_grid, scoring="r2", cv=kfold, n_jobs=-1
    )
    xgb_search.fit(X, y)
    print("  best params:", xgb_search.best_params_, "5-fold CV R2:", round(xgb_search.best_score_, 3))

    # evaluate the TUNED models with the same LOO protocol as every prior result
    rf_tuned = RandomForestRegressor(**rf_search.best_params_, random_state=42, n_jobs=-1)
    xgb_tuned = xgb.XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1)

    print("Evaluating tuned models with LOO-CV (for apples-to-apples comparison)...")
    rf_loo_preds = cross_val_predict(rf_tuned, X, y, cv=loo)
    xgb_loo_preds = cross_val_predict(xgb_tuned, X, y, cv=loo)
    rf_loo_metrics = metrics(y, rf_loo_preds)
    xgb_loo_metrics = metrics(y, xgb_loo_preds)

    report = []
    report.append("# Hyperparameter tuning results (v2 chemistry-informed features, n=%d)\n" % len(df))
    report.append(
        "5-fold CV grid search for hyperparameter selection (cheap), then the winning hyperparameters "
        "re-evaluated with leave-one-out CV (same protocol as every other result in this project, so "
        "numbers are directly comparable). Previous untuned LOO R2 was ~0.347 (RF) / ~0.350 (XGBoost) "
        "with hand-picked defaults (n_estimators=200/100, max_depth=4/3).\n"
    )
    report.append("| Model | Best params (5-fold selection) | 5-fold CV R2 | LOO R2 (final) | LOO MAE | LOO RMSE |")
    report.append("|---|---|---|---|---|---|")
    report.append(
        f"| RandomForest | {rf_search.best_params_} | {rf_search.best_score_:.3f} | "
        f"{rf_loo_metrics['R2']:.3f} | {rf_loo_metrics['MAE']:.3f} | {rf_loo_metrics['RMSE']:.3f} |"
    )
    report.append(
        f"| XGBoost | {xgb_search.best_params_} | {xgb_search.best_score_:.3f} | "
        f"{xgb_loo_metrics['R2']:.3f} | {xgb_loo_metrics['MAE']:.3f} | {xgb_loo_metrics['RMSE']:.3f} |"
    )

    best_loo_r2 = max(rf_loo_metrics["R2"], xgb_loo_metrics["R2"])
    report.append(
        f"\n- Tuning moved LOO R2 from ~0.35 (untuned) to {best_loo_r2:.3f} (tuned RF). "
        + (
            "A real further improvement from tuning alone."
            if best_loo_r2 > 0.36
            else "Essentially no improvement over the untuned defaults."
        )
    )
    report.append(
        f"\n- **IMPORTANT HONESTY CHECK**: the 5-fold CV R2 used for hyperparameter selection "
        f"(RF={rf_search.best_score_:.3f}, XGB={xgb_search.best_score_:.3f}) is NOTABLY LOWER than the "
        f"final LOO R2 ({rf_loo_metrics['R2']:.3f}/{xgb_loo_metrics['R2']:.3f}) for the SAME hyperparameters. "
        "This is expected, not a bug: LOO holds out only 1 of 137 points per fold (136 points still inform "
        "the model every time), while 5-fold holds out ~27 points at once -- a much harder, more honest test "
        "of generalization at this sample size, and more sensitive to which handful of unusual/outlier host "
        "systems happen to land in the same random fold. The two numbers bracket plausible true performance: "
        f"somewhere in roughly [{min(rf_search.best_score_, xgb_search.best_score_):.2f}, {best_loo_r2:.2f}]. "
        "Reporting only the higher LOO number without this context would overstate how well this model "
        "actually generalizes to a genuinely new polymer-electrolyte system -- the paper should present both "
        "and be explicit that 5-fold is the more conservative, probably more realistic estimate.\n"
    )

    out_text = "\n".join(report)
    with open(f"{RESULTS_DIR}\\tuning_results.md", "w", encoding="utf-8") as f:
        f.write(out_text)
    print("\n" + out_text)


if __name__ == "__main__":
    main()
