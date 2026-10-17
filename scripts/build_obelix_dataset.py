"""Parse OBELiX processed.csv into a clean, feature-engineered table for ML.

OBELiX is a Li-ion inorganic solid-electrolyte crystal dataset. It is used here
as a large (n~599), well-curated benchmark to validate the ML pipeline
(composition/structure -> log10 ionic conductivity) before applying the same
methodology to the much smaller, novel multivalent polymer-electrolyte dataset.
"""
import re
import numpy as np
import pandas as pd

SRC = r"C:\Polyemr Electrolyter ML project\data\raw\obelix\OBELiX-main\data\processed.csv"
OUT = r"C:\Polyemr Electrolyter ML project\data\processed\obelix_features.csv"

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)([0-9]+\.?[0-9]*)")


def parse_composition(formula):
    counts = {}
    for el, num in ELEMENT_RE.findall(formula):
        counts[el] = counts.get(el, 0.0) + float(num)
    return counts


def main():
    df = pd.read_csv(SRC)
    df = df.dropna(subset=["Ionic conductivity (S cm-1)", "Composition"])
    df = df[df["Ionic conductivity (S cm-1)"] > 0]

    comps = df["Composition"].apply(parse_composition)
    all_elements = sorted(set(el for c in comps for el in c))

    feat = pd.DataFrame(index=df.index)
    for el in all_elements:
        feat[f"n_{el}"] = comps.apply(lambda c: c.get(el, 0.0))

    total_atoms = feat.sum(axis=1)
    feat["total_atoms"] = total_atoms
    feat["n_distinct_elements"] = (feat[[f"n_{el}" for el in all_elements]] > 0).sum(axis=1)
    feat["li_fraction"] = feat["n_Li"] / total_atoms.replace(0, np.nan)

    feat["space_group_number"] = df["Space group number"].values
    for col in ["a", "b", "c", "alpha", "beta", "gamma"]:
        feat[col] = df[col].values
    # crude unit-cell volume proxy (not a proper triclinic volume formula, but a
    # monotonic size proxy useful as an engineered feature)
    feat["lattice_volume_proxy"] = feat["a"] * feat["b"] * feat["c"]

    feat["log10_conductivity"] = np.log10(df["Ionic conductivity (S cm-1)"].values)
    feat["composition"] = df["Composition"].values
    feat["doi"] = df["DOI"].values

    feat = feat.dropna()
    feat.to_csv(OUT, index=False)
    print(f"Wrote {len(feat)} rows x {feat.shape[1]} cols -> {OUT}")
    print("Target stats (log10 S/cm):", feat["log10_conductivity"].describe().to_dict())


if __name__ == "__main__":
    main()
