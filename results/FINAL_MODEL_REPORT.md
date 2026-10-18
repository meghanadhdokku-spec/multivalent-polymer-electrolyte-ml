# FINAL MODEL REPORT -- multivalent (Mg/Zn/Na) polymer electrolyte conductivity

This supersedes model_results.md / tuning_results.md / nested_cv_results.md as the single reference. Those files remain as the record of how this result was reached (dataset growth -> feature engineering -> hyperparameter tuning -> leakage check).

## Bottom line

- **Final model: RandomForestRegressor({'n_estimators': 400, 'max_depth': 6, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}), trained on 137 literature-mined Mg/Zn/Na polymer-electrolyte data points.**
- **Validated performance (leave-one-out CV): R2 = 0.450**, vs a mean-predicting dummy baseline R2 = -0.015. MAE = 0.889 log10(S/cm) (i.e. typical prediction is within roughly a factor of 7.7x of the true conductivity).
- **Caveat carried forward honestly**: a stricter 5-fold CV estimate (computed during hyperparameter selection, see tuning_results.md) gave R2 = 0.272 -- notably lower. The gap is best explained by 5-fold's higher variance at n=137 (holding out ~27 points at once vs 1 for LOO), not by leakage (nested_cv.py's fully unbiased check gave R2=0.451, matching this result, not the 5-fold one). **Report both numbers in the paper**: ~0.45 as the headline, ~0.27 as a deliberately conservative lower bound.

- For context, the SAME pipeline on OBELiX (n=599, Li-ion inorganic crystals, a mature benchmark with clean/consistent measurements) reaches R2=0.79. The gap between that and ~0.45 here is the honest measure of how much harder -- and how much more data-starved -- the multivalent/polymer problem is, which is the paper's actual point.


## What the model is built on

- 137 rows across 3 cations (Mg, Zn, Na) and 106 distinct polymer-host systems, hand-curated from the literature across several rounds of targeted search and dataset growth. 2 alkaline/KOH-soak rows were excluded (they measure OH- transport, not the target cation).

- Features (chemistry-informed, not just coarse categorical flags): numeric salt concentration where parseable (51/137 rows), anion identity (one-hot, rare categories with <=3 members pooled into 'rare_other') + a fluorinated-anion flag, cation physical descriptors (ionic radius, charge, charge density -- currently redundant with the one-hot cation flags at only 3 cations, would matter more with a 4th added later), expanded polymer-host chemistry flags (ether-oxygen backbone, natural biopolymer, fluorinated backbone, hydrogel/aqueous), plasticizer-type flags (ionic liquid / ceramic filler / organic co-solvent), and a secondary-source/low-confidence flag.


## Feature importances (top 15)

| Feature | Importance |
|---|---|
| cation_Zn | 0.235 |
| host_has_ether_oxygen | 0.093 |
| plasticizer_none | 0.066 |
| temperature_C | 0.066 |
| host_has_PEO | 0.061 |
| host_is_hydrogel_aqueous | 0.051 |
| value_is_approximate | 0.051 |
| anion_none_single_ion | 0.045 |
| is_secondary_source | 0.031 |
| salt_wt_pct_imputed | 0.026 |
| temperature_assumed | 0.024 |
| anion_fluorinated | 0.022 |
| salt_wt_pct | 0.020 |
| anion_triflate | 0.016 |
| host_is_blend | 0.015 |

## Known limitations

- **Sample size**: 137 rows across 106-ish distinct host systems is small for 46 features; most host systems appear only once or twice, so the model leans heavily on cation identity and a few coarse host-chemistry flags rather than learning fine structure-property trends.
- **Measurement heterogeneity**: different papers, different temperature conventions (many 'assumed RT'), different confidence levels (several rows are secondary-sourced from review tables, not independently verified against primary full text) -- this is irreducible noise that caps achievable R2 regardless of model or feature choices.
- **LOO vs 5-fold discrepancy**: see Bottom Line above -- report both numbers, don't cherry-pick.
- **Concentration coverage**: only 51/137 rows have a parseable numeric salt concentration; the rest are median-imputed with a flag. More systematic concentration extraction would likely help more than further hyperparameter tuning at this point.
- **Growth has diminishing returns**: several rounds of literature search have mined essentially every efficient review-table source found; further rows would mostly come from individual-paper search (expensive per row) rather than the efficient review-table approach that drove most of this growth.


## Not yet done

- No figures/plots, no manuscript drafting -- this stage covered dataset curation and modeling only.
