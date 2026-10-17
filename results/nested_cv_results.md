# Nested cross-validation results (v2 features, n=137)

Unbiased estimate: for every leave-one-out outer fold, hyperparameters are re-selected via inner 5-fold CV using ONLY that fold's training data (the held-out point never influences hyperparameter choice). This corrects the validity gap in tuning_results.md, where hyperparameters were chosen using the full dataset before 'testing' on the full dataset.

| Model | Nested-CV R2 | MAE | RMSE |
|---|---|---|---|
| RandomForest | 0.451 | 0.886 | 1.160 |
| XGBoost | 0.382 | 0.938 | 1.231 |

- **Correction to the hypothesis stated in this script's own comments before running it**: nested-CV R2 (0.451) came out essentially IDENTICAL to the "leaked" LOO estimate from tuning_results.md (0.449), NOT close to the lower 5-fold-only estimate (0.272) as predicted. The hyperparameter-selection leakage theorized earlier turned out NOT to meaningfully bias the result in this case -- likely because the best hyperparameters are fairly stable across different 136-point training subsets at this sample size, so inner-fold-specific tuning converges to similar choices as whole-dataset tuning anyway. This is worth stating plainly rather than quietly editing away: the leakage concern was a reasonable thing to check, it just didn't turn out to matter here.
- So what DOES explain the gap between 0.45 (both LOO variants) and 0.27 (5-fold)? Most likely 5-fold's own higher variance at n=137: holding out ~27 points at once means a single unlucky split (e.g. several of the harder/more unusual host systems landing in the same test fold) can swing the score substantially, whereas LOO averages over 137 individual single-point tests and is more stable (if also structurally more forgiving, per the earlier near-duplicate-host caveat). Neither number is "wrong"; they're answering slightly different questions about generalization at small n.
- **Recommended number to report in the paper**: ~0.45 (RandomForest), since both the leaked and properly-nested estimates agree on it, but report the 5-fold figure (~0.27) alongside it as a deliberately conservative lower bound, with this discrepancy explained as above -- that transparency is more defensible than picking either number alone.
