# Multivalent Polymer Electrolyte ML Dataset and Pipeline

Data, code, and results accompanying "Machine Learning for Ionic Conductivity Prediction in Polymer Electrolytes: A Benchmark on Lithium Systems and a Feasibility Study on the Data-Scarce Multivalent (Mg, Zn, Na) Frontier."

## What this is

A literature-mined dataset of 137 experimentally measured ionic conductivities for magnesium-, zinc-, and sodium-conducting polymer electrolytes (135 after excluding 2 non-target-cation rows, the first of its kind, to our knowledge), plus the full feature-engineering and modeling pipeline used to benchmark it against a published lithium-ion dataset (OBELiX).

## Contents

- **`data/`**
  - `multivalent_literature_extract_merged_v3.csv` — the raw, literature-mined dataset: 137 rows (including 2 alkaline rows excluded from modeling, kept here for transparency), with per-row composition, salt, plasticizer/filler, measured conductivity, temperature, and source citation.
  - `multivalent_features_v2.csv` — the engineered feature table (anion identity, cation physical descriptors, host/plasticizer chemistry classes) used to train the final model: 135 rows.
  - `obelix_features.csv` — engineered features for the OBELiX lithium benchmark, derived from the published [OBELiX dataset](https://github.com/NRC-Mila/OBELiX) (not redistributed here — see that repository for the original data and its own license).
  - `supplementary_table_S1.md` — the 85 distinct literature sources behind the multivalent dataset, with per-source confidence flags.
- **`scripts/`** — the pipeline in run order: `build_obelix_dataset.py`, `build_multivalent_dataset_v2.py`, `build_supplementary_table.py`, `train_models.py`, `tune_models.py`, `nested_cv.py`, `final_model.py`, `generate_figures.py`.
- **`results/`** — `final_model.pkl` (the trained Random Forest model artifact), `FINAL_MODEL_REPORT.md` (the single authoritative results summary), and the intermediate result files the final numbers were derived from.
- **`figures/`** — the 5 figures used in the manuscript.

## Reproducing the results

Requires Python 3.12+ with `pandas`, `numpy`, `scikit-learn`, `xgboost`, and `matplotlib`. Run the scripts in `scripts/` in the order listed above; each reads from and writes to the `data/`/`results/` paths referenced in this README (adjust paths at the top of each script if your local layout differs from this bundle's).

## Known data-quality notes (documented, not hidden)

- 56 of 135 modeled rows (41%) are secondary-sourced (read from a review paper's comparison table or a search-engine abstract snippet, not independently verified against primary full text) — flagged per-row via the `notes` column and the `is_secondary_source` feature flag. At the source level (one flag per distinct citation, in `supplementary_table_S1.md`), 24 of 85 sources (28%) are flagged low-confidence.
- 2 rows describing alkaline (KOH-soak) systems were excluded from modeling since they measure hydroxide-ion transport, not the labeled cation's transport — see `supplementary_table_S1.md` for details.
- Salt concentration is directly available for only 52 of 135 modeled rows; the rest are median-imputed with an explicit flag.

## Citation

If you use this dataset or pipeline, please cite the accompanying paper [citation details to be added once published / assigned a DOI].

## License

MIT (see `LICENSE`). Code and data are both released under this license.
