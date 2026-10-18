# Hyperparameter tuning results (v2 chemistry-informed features, n=135)

5-fold CV grid search for hyperparameter selection (cheap), then the winning hyperparameters re-evaluated with leave-one-out CV (same protocol as every other result in this project, so numbers are directly comparable). Previous untuned LOO R2 was ~0.347 (RF) / ~0.350 (XGBoost) with hand-picked defaults (n_estimators=200/100, max_depth=4/3).

| Model | Best params (5-fold selection) | 5-fold CV R2 | LOO R2 (final) | LOO MAE | LOO RMSE |
|---|---|---|---|---|---|
| RandomForest | {'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 100} | 0.359 | 0.469 | 0.861 | 1.146 |
| XGBoost | {'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 50} | 0.405 | 0.435 | 0.871 | 1.182 |

- Tuning moved LOO R2 from ~0.35 (untuned) to 0.469 (tuned RF). A real further improvement from tuning alone.

- **IMPORTANT HONESTY CHECK**: the 5-fold CV R2 used for hyperparameter selection (RF=0.359, XGB=0.405) is NOTABLY LOWER than the final LOO R2 (0.469/0.435) for the SAME hyperparameters. This is expected, not a bug: LOO holds out only 1 of 137 points per fold (136 points still inform the model every time), while 5-fold holds out ~27 points at once -- a much harder, more honest test of generalization at this sample size, and more sensitive to which handful of unusual/outlier host systems happen to land in the same random fold. The two numbers bracket plausible true performance: somewhere in roughly [0.36, 0.47]. Reporting only the higher LOO number without this context would overstate how well this model actually generalizes to a genuinely new polymer-electrolyte system -- the paper should present both and be explicit that 5-fold is the more conservative, probably more realistic estimate.
