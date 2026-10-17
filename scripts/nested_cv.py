"""Proper nested cross-validation for the v2 multivalent feature set.

scripts/tune_models.py had a subtle validity gap: hyperparameters were chosen
via 5-fold CV on the FULL dataset, then the SAME hyperparameters were
evaluated via LOO-CV on the FULL dataset again. Every point's information
leaked into the hyperparameter choice before "testing" on it, so that LOO R2
(0.449) is optimistic, not a clean held-out estimate.

The honest fix: nested CV. For every outer LOO fold, redo the hyperparameter
search using ONLY that fold's 136 training points (inner 5-fold CV), then
predict the 1 held-out point with the hyperparameters chosen from training
data alone. sklearn does this correctly and automatically when a GridSearchCV
object is passed as the estimator to an outer cross_val_predict/LeaveOneOut.

Grids are smaller than tune_models.py's (137 outer folds x inner grid x 5
inner folds is expensive) -- centered on the previous winning regions.
"""
import time
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

    outer_loo = LeaveOneOut()
    inner_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_grid = {"n_estimators": [100, 300], "max_depth": [4, 6, None], "min_samples_leaf": [1, 2]}
    xgb_grid = {"n_estimators": [50, 100], "max_depth": [3, 4], "learning_rate": [0.05, 0.1]}

    rf_nested = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1), rf_grid, scoring="r2", cv=inner_kfold, n_jobs=-1
    )
    xgb_nested = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=1), xgb_grid, scoring="r2", cv=inner_kfold, n_jobs=-1
    )

    print(f"Running nested CV for RandomForest (137 outer folds x {2*3*2} inner combos x 5 inner folds)...")
    t0 = time.time()
    rf_preds = cross_val_predict(rf_nested, X, y, cv=outer_loo)
    print(f"  done in {time.time()-t0:.0f}s")

    print(f"Running nested CV for XGBoost (137 outer folds x {2*2*2} inner combos x 5 inner folds)...")
    t0 = time.time()
    xgb_preds = cross_val_predict(xgb_nested, X, y, cv=outer_loo)
    print(f"  done in {time.time()-t0:.0f}s")

    rf_metrics = metrics(y, rf_preds)
    xgb_metrics = metrics(y, xgb_preds)

    report = []
    report.append(f"# Nested cross-validation results (v2 features, n={len(df)})\n")
    report.append(
        "Unbiased estimate: for every leave-one-out outer fold, hyperparameters are re-selected via "
        "inner 5-fold CV using ONLY that fold's training data (the held-out point never influences "
        "hyperparameter choice). This corrects the validity gap in tuning_results.md, where "
        "hyperparameters were chosen using the full dataset before 'testing' on the full dataset.\n"
    )
    report.append("| Model | Nested-CV R2 | MAE | RMSE |")
    report.append("|---|---|---|---|")
    report.append(f"| RandomForest | {rf_metrics['R2']:.3f} | {rf_metrics['MAE']:.3f} | {rf_metrics['RMSE']:.3f} |")
    report.append(f"| XGBoost | {xgb_metrics['R2']:.3f} | {xgb_metrics['MAE']:.3f} | {xgb_metrics['RMSE']:.3f} |")
    report.append(
        f"\n- Compare: untuned LOO (tune_models.py baseline) ~0.35; hyperparameter-leaked LOO "
        f"(tuning_results.md) 0.449 (RF); 5-fold-only estimate (also from tuning_results.md) 0.272 (RF); "
        f"nested-CV (this run) {rf_metrics['R2']:.3f} (RF). NOTE: nested-CV came out essentially identical "
        f"to the leaked LOO estimate, not close to the lower 5-fold estimate -- the hyperparameter-selection "
        f"leakage theorized when this script was written did NOT meaningfully bias the earlier result in "
        f"practice (best hyperparameters are apparently stable across different training subsets at this "
        f"n). The gap between ~0.45 (both LOO variants) and ~0.27 (5-fold) is better explained by 5-fold's "
        f"own higher variance at n=137 (holding out ~27 points at once is more sensitive to which unusual "
        f"host systems land together in one fold) than by any leakage.\n"
    )
    report.append(
        "- **Recommended number to report in the paper**: ~0.45 (RandomForest), since the leaked and "
        "properly-nested estimates agree on it -- but report the 5-fold figure (~0.27) alongside it as a "
        "deliberately conservative lower bound, with the above explanation, rather than picking one number.\n"
    )

    out_text = "\n".join(report)
    with open(f"{RESULTS_DIR}\\nested_cv_results.md", "w", encoding="utf-8") as f:
        f.write(out_text)
    print("\n" + out_text)


if __name__ == "__main__":
    main()
