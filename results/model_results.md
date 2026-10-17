# Model training results

## Dataset A: OBELiX (Li-ion inorganic crystals, n=599)

80/20 train/test split, random_state=42. Target: log10(S/cm).

| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Dummy(mean) | -0.003 | 2.674 | 3.678 |
| RandomForest | 0.770 | 0.987 | 1.761 |
| XGBoost | 0.786 | 0.961 | 1.699 |

Features used (66): n_Ag, n_Al, n_B, n_Ba, n_Bi, n_Br, n_C, n_Ca, n_Cd, n_Ce, n_Cl, n_Co, n_Cr, n_Cu, n_Er, n_F, n_Fe, n_Ga, n_Gd, n_Ge, n_H, n_Hf, n_I, n_In, n_K, n_La, n_Li, n_Lu, n_Mg, n_Mn, n_Mo, n_N, n_Na, n_Nb, n_Nd, n_O, n_P, n_Pb, n_Pr, n_S, n_Sb, n_Sc, n_Se, n_Si, n_Sm, n_Sn, n_Sr, n_Ta, n_Te, n_Ti, n_V, n_W, n_Y, n_Zn, n_Zr, total_atoms, n_distinct_elements, li_fraction, space_group_number, a, b, c, alpha, beta, gamma, lattice_volume_proxy


## Dataset B: Multivalent (Mg/Zn/Na) polymer electrolytes, literature extract (n=137)

Leave-one-out cross-validation (n is far too small for a held-out split). Target: log10(S/cm).

| Model | R2 (LOO) | MAE | RMSE |
|---|---|---|---|
| Dummy(mean) | -0.015 | 1.250 | 1.577 |
| RandomForest | 0.330 | 0.985 | 1.282 |
| XGBoost | 0.274 | 1.026 | 1.334 |

Features used (14): cation_Mg, cation_Zn, cation_Na, has_plasticizer_or_filler, temperature_C, temperature_assumed, value_is_approximate, host_has_PVA, host_has_PEO, host_has_PVDF, host_has_cellulose, host_has_chitosan, host_has_nafion, host_is_blend


## Interpretation

- OBELiX results validate that the pipeline (composition/structure -> log10 conductivity, RF/XGBoost vs. a mean-predicting dummy baseline) works and that tree ensembles beat the baseline on an established, well-populated inorganic Li-ion dataset.

- The multivalent (Mg/Zn/Na) polymer dataset is the paper's actual novel contribution. Tree models beat the dummy baseline under LOO CV (R2 ~0.27-0.33), but this should NOT be read as a robust result: several rows share the same polymer host measured at different temperatures only (e.g. PAES-g-PEG, PVA/HPC, Nafion, PVA-borax-glycerol at 3 temperatures), so in LOO CV the model can partly recognize a left-out row's host fingerprint from its near-duplicate still in the training fold -- this inflates R2 versus a true held-out-composition test. At n=137 across only 106 distinct host systems this cannot be fully fixed by re-splitting; it should be reported as a caveat, and the real headline finding is the data-scarcity problem itself (137 usable points, 18 of them only order-of-magnitude, across 3 cations and 106 distinct host systems), not the R2 number.

- Framing for the paper: Part 1 (OBELiX) demonstrates the ML methodology on a mature benchmark; Part 2 (multivalent) applies the same methodology to a hand-curated novel dataset and the small-n result itself supports the paper's stated novelty case -- that public ML-ready data for multivalent/non-lithium polymer electrolytes barely exists yet.

- TREND ACROSS GROWTH PASSES (worth reporting explicitly): LOO R2 for the tree models went ~0.47-0.49 (n=48, 40 distinct hosts) -> ~0.40-0.41 (n=91, 75 distinct hosts) -> ~0.27-0.33 (n=137, 106 distinct hosts). This monotonic decline as the dataset grew more diverse is itself informative: it shows the earlier, higher R2 values were substantially inflated by LOO CV recognizing near-duplicate host fingerprints, not by the model learning real structure-conductivity relationships. The current, more diverse dataset gives a more honest (lower) estimate of what coarse categorical features can predict, and argues that further improvement needs richer chemical features (numeric salt concentration, molecular descriptors, anion identity), not just more rows of the same coarse feature types.


## Dataset B-v2: Multivalent, chemistry-informed features (n=137)

Same rows as Dataset B, but with richer features: numeric salt concentration (wt%/M/mol%, median-imputed with an imputed-flag), anion identity one-hot + fluorinated-anion flag, cation physical descriptors (Shannon ionic radius, charge, charge density z/r) instead of relying only on one-hot cation identity, expanded host chemistry flags (ether-oxygen backbone, natural biopolymer, fluorinated backbone, hydrogel/aqueous), plasticizer TYPE flags (ionic liquid, ceramic filler, organic co-solvent) instead of one binary, and a secondary-source/low-confidence flag parsed from notes. Leave-one-out CV, same protocol as Dataset B. Target: log10(S/cm).

| Model | R2 (LOO) | MAE | RMSE |
|---|---|---|---|
| Dummy(mean) | -0.015 | 1.250 | 1.577 |
| RandomForest | 0.347 | 0.924 | 1.265 |
| XGBoost | 0.350 | 0.959 | 1.263 |

Features used (47): cation_Mg, cation_Zn, cation_Na, cation_ionic_radius_A, cation_charge, cation_charge_density, anion_fluorinated, salt_wt_pct, salt_molarity_M, salt_mol_pct, concentration_known, temperature_C, temperature_assumed, value_is_approximate, host_has_PVA, host_has_PEO, host_has_PVDF, host_has_cellulose, host_has_chitosan, host_has_nafion, host_is_blend, host_has_ether_oxygen, host_is_natural_biopolymer, host_has_fluorine, host_is_hydrogel_aqueous, plasticizer_ionic_liquid, plasticizer_ceramic_filler, plasticizer_organic_cosolvent, plasticizer_none, is_secondary_source, anion_FSI, anion_TFSI, anion_acetate, anion_bromide, anion_chloride, anion_iodide, anion_nitrate, anion_none_single_ion, anion_other_unknown, anion_perchlorate, anion_sulfate, anion_tetrafluoroborate, anion_thiocyanate, anion_triflate, salt_wt_pct_imputed, salt_molarity_M_imputed, salt_mol_pct_imputed


RandomForest feature importances (top 15):

| Feature | Importance |
|---|---|
| cation_Zn | 0.266 |
| anion_none_single_ion | 0.096 |
| host_has_ether_oxygen | 0.093 |
| host_has_PEO | 0.088 |
| host_is_hydrogel_aqueous | 0.069 |
| value_is_approximate | 0.044 |
| temperature_C | 0.042 |
| plasticizer_none | 0.040 |
| anion_chloride | 0.035 |
| salt_wt_pct_imputed | 0.031 |
| is_secondary_source | 0.018 |
| salt_wt_pct | 0.017 |
| temperature_assumed | 0.014 |
| anion_fluorinated | 0.012 |
| host_has_fluorine | 0.011 |

- **v1 vs v2 comparison**: coarse-feature R2 was ~0.27-0.33; chemistry-informed v2 R2 is ~0.35-0.35. Feature engineering HELPED, supporting the hypothesis that richer chemical features (not just more rows) are the better investment at this dataset size -- see the feature importances above for which new features the model actually leaned on.

- Caveat on the cation physical descriptors (ionic radius, charge, charge density): with only 3 cations in this dataset, these are deterministic functions of which one-hot cation flag is set (e.g. cation_Mg=1 always implies radius=0.72, charge=2) -- they carry no information beyond the one-hot flags yet, which is why they don't show up with independent importance above. They would only start earning their keep if a 4th+ cation (e.g. Ca2+) were added, letting the model learn a continuous radius/charge-density trend rather than memorizing 3 separate categories.
