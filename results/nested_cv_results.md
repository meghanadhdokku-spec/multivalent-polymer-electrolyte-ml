# Nested cross-validation results (v2 features, n=135)

Unbiased estimate: for every leave-one-out outer fold, hyperparameters are re-selected via inner 5-fold CV using ONLY that fold's training data (the held-out point never influences hyperparameter choice). This corrects the validity gap in tuning_results.md, where hyperparameters were chosen using the full dataset before 'testing' on the full dataset.

| Model | Nested-CV R2 | MAE | RMSE |
|---|---|---|---|
| RandomForest | 0.478 | 0.862 | 1.136 |
| XGBoost | 0.463 | 0.849 | 1.152 |

- Compare: untuned LOO (tune_models.py baseline) ~0.35; hyperparameter-leaked LOO (tuning_results.md) 0.449 (RF); 5-fold-only estimate (also from tuning_results.md) 0.272 (RF); nested-CV (this run) 0.478 (RF). NOTE: nested-CV came out essentially identical to the leaked LOO estimate, not close to the lower 5-fold estimate -- the hyperparameter-selection leakage theorized when this script was written did NOT meaningfully bias the earlier result in practice (best hyperparameters are apparently stable across different training subsets at this n). The gap between ~0.45 (both LOO variants) and ~0.27 (5-fold) is better explained by 5-fold's own higher variance at n=137 (holding out ~27 points at once is more sensitive to which unusual host systems land together in one fold) than by any leakage.

- **Recommended number to report in the paper**: ~0.45 (RandomForest), since the leaked and properly-nested estimates agree on it -- but report the 5-fold figure (~0.27) alongside it as a deliberately conservative lower bound, with the above explanation, rather than picking one number.
