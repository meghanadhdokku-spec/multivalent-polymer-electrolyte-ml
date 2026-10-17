"""Feature engineering v2 for the multivalent (Mg/Zn/Na) polymer electrolyte dataset.

v1 (build_multivalent_dataset.py) used only coarse categorical flags (cation
one-hot, 7 binary host-family flags, a plasticizer binary, temperature). This
version adds chemistry-informed features that actually encode *why* a system
should conduct better or worse, instead of just *which bucket* it falls in:

  - numeric salt concentration (wt%, molarity) parsed where possible
  - anion identity (perchlorate/triflate/TFSI/nitrate/...) + a fluorinated-anion
    flag (fluorinated, weakly-coordinating anions are known to dissociate more
    easily and should correlate with higher conductivity)
  - cation physical descriptors (Shannon ionic radius, charge, charge density
    z/r) instead of relying only on one-hot cation identity -- this lets the
    model relate conductivity to the actual physical reason multivalent
    cations behave differently (higher charge density -> stronger binding to
    polymer donor atoms -> usually lower mobility), which also ties directly
    into the paper's own novelty framing
  - expanded host chemistry flags (ether-oxygen backbone, natural biopolymer,
    fluorinated backbone, hydrogel/aqueous system)
  - plasticizer/filler TYPE flags (ionic liquid, ceramic filler, organic
    co-solvent) instead of one binary "has plasticizer"
  - a data-quality / secondary-source flag parsed from the notes column
"""
import re
import numpy as np
import pandas as pd

SRC = r"C:\Polyemr Electrolyter ML project\data\raw\multivalent_literature_extract_merged_v3.csv"
OUT = r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv"

# Shannon effective ionic radii, 6-coordinate, Angstrom (Shannon 1976)
CATION_RADIUS_A = {"Mg": 0.72, "Zn": 0.74, "Na": 1.02}
CATION_CHARGE = {"Mg": 2, "Zn": 2, "Na": 1}


def parse_conductivity(raw):
    s = str(raw).strip()
    is_approx = "~" in s or "order of magnitude" in s or "range" in s
    nums = re.findall(r"[-+]?\d*\.?\d+[Ee][-+]?\d+", s)
    if not nums:
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
    nums = [float(n) for n in nums if n not in ("",)]
    if not nums:
        return np.nan, True
    val = 10 ** (np.mean([np.log10(n) for n in nums])) if len(nums) >= 2 else nums[0]
    return val, is_approx


def parse_temperature(raw):
    s = str(raw).strip()
    if "RT" in s:
        return 25.0, True
    m = re.search(r"[-+]?\d+\.?\d*", s)
    if m:
        return float(m.group()), False
    return np.nan, True


def parse_concentration(raw):
    s = str(raw).strip().lower()
    wt_pct = np.nan
    molarity = np.nan
    m = re.search(r"(\d+\.?\d*)\s*wt\s*%", s)
    if m:
        wt_pct = float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*m\b(?!ol)", s)  # "1.0 M" but not "mol%"
    if m:
        molarity = float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*mol\s*%", s)
    mol_pct = float(m.group(1)) if m else np.nan
    known = int(not (np.isnan(wt_pct) and np.isnan(molarity) and np.isnan(mol_pct)))
    return wt_pct, molarity, mol_pct, known


def classify_anion(salt_text):
    s = str(salt_text).lower()
    if "tfsi" in s:
        return "TFSI", True
    if "fsi" in s:
        return "FSI", True
    if "clo4" in s or "perchlorate" in s:
        return "perchlorate", False
    if "cf3so3" in s or "otf" in s or "triflate" in s:
        return "triflate", True
    if "pf6" in s or "hexafluorophosphate" in s:
        return "hexafluorophosphate", True
    if "bf4" in s or "tetrafluoroborate" in s:
        return "tetrafluoroborate", True
    if "no3" in s or "nitrate" in s:
        return "nitrate", False
    if "so4" in s or "sulfate" in s:
        return "sulfate", False
    if "scn" in s or "thiocyanate" in s:
        return "thiocyanate", False
    if "acetate" in s or "ch3coo" in s:
        return "acetate", False
    if "koh" in s or "hydroxide" in s:
        return "hydroxide", False
    if re.search(r"\bcl\d*\b", s) or "chloride" in s or "cl2" in s:
        return "chloride", False
    if re.search(r"\bbr\d*\b", s) or "bromide" in s:
        return "bromide", False
    if re.search(r"\bnai\b", s) or "iodide" in s:
        return "iodide", False
    if "none" in s or "single-ion" in s or "single ion" in s:
        return "none_single_ion", False
    return "other_unknown", False


def host_flags(host):
    h = str(host).lower()
    return {
        "host_has_PVA": int("pva" in h or "poly(vinyl alcohol)" in h),
        "host_has_PEO": int("peo" in h or "poly(ethylene oxide)" in h or "pec" in h),
        "host_has_PVDF": int("pvdf" in h or "vinylidene" in h or "vdf" in h),
        "host_has_cellulose": int("cellulose" in h or "hpmc" in h or "hec" in h or "cmc" in h),
        "host_has_chitosan": int("chitosan" in h),
        "host_has_nafion": int("nafion" in h),
        "host_is_blend": int("blend" in h or "/" in h or "+" in h or ":" in h),
        "host_has_ether_oxygen": int(
            any(k in h for k in ["peo", "peg", "pec", "ether", "pthf", "ethylene oxide"])
        ),
        "host_is_natural_biopolymer": int(
            any(
                k in h
                for k in [
                    "chitosan", "cellulose", "alginate", "pullulan", "carrageenan",
                    "agar", "guar", "dextran", "starch", "pectin",
                ]
            )
        ),
        "host_has_fluorine": int(
            any(k in h for k in ["pvdf", "vinylidene", "vdf", "nafion", "trfe", "ctfe"])
        ),
        "host_is_hydrogel_aqueous": int(
            any(k in h for k in ["hydrogel", "gel", "aqueous"])
        ),
    }


def plasticizer_flags(text):
    t = str(text).lower()
    return {
        "plasticizer_ionic_liquid": int(
            any(k in t for k in ["ionic liquid", "emitf", "emimbf4", "emi"])
        ),
        "plasticizer_ceramic_filler": int(
            any(
                k in t
                for k in ["al2o3", "mgo", "tio2", "sio2", "nasicon", "nzsp", "batio3", "ceramic", "filler"]
            )
        ),
        "plasticizer_organic_cosolvent": int(
            any(k in t for k in ["ec/pc", "ethylene carbonate", "propylene carbonate", "dmso", "glycerol", "succinonitrile", "des", "eutectic"])
        ),
        "plasticizer_none": int(t.strip() in ("none", "nan", "")),
    }


def classify_quality(notes):
    n = str(notes).lower()
    return {
        "is_secondary_source": int(
            any(k in n for k in ["secondary source", "not independently confirmed", "unconfirmed", "low confidence", "low-confidence"])
        ),
    }


def main():
    df = pd.read_csv(SRC)
    df = df.dropna(subset=["ionic_conductivity_S_per_cm", "cation_type"])

    is_alkaline = df["salt_or_dopant"].astype(str).str.contains("KOH", case=False, na=False)
    print(f"Excluding {is_alkaline.sum()} alkaline (KOH) row(s)")
    df = df[~is_alkaline].reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        cond, is_approx = parse_conductivity(r["ionic_conductivity_S_per_cm"])
        if np.isnan(cond) or cond <= 0:
            continue
        temp, temp_assumed = parse_temperature(r["temperature_C"])
        if np.isnan(temp):
            temp = 25.0
            temp_assumed = True

        wt_pct, molarity, mol_pct, conc_known = parse_concentration(r["salt_concentration_or_ratio"])
        anion, anion_fluorinated = classify_anion(r["salt_or_dopant"])
        cation = r["cation_type"]

        row = {
            "cation_Mg": int(cation == "Mg"),
            "cation_Zn": int(cation == "Zn"),
            "cation_Na": int(cation == "Na"),
            "cation_ionic_radius_A": CATION_RADIUS_A.get(cation, np.nan),
            "cation_charge": CATION_CHARGE.get(cation, np.nan),
            "cation_charge_density": CATION_CHARGE.get(cation, np.nan) / CATION_RADIUS_A.get(cation, np.nan),
            "anion_fluorinated": int(anion_fluorinated),
            "salt_wt_pct": wt_pct,
            "salt_molarity_M": molarity,
            "salt_mol_pct": mol_pct,
            "concentration_known": conc_known,
            "temperature_C": temp,
            "temperature_assumed": int(temp_assumed),
            "value_is_approximate": int(is_approx),
            "notes_text": str(r.get("notes", "")),
        }
        row.update(host_flags(r["polymer_host"]))
        row.update(plasticizer_flags(r["plasticizer_or_filler"]))
        row.update(classify_quality(r.get("notes", "")))
        row["log10_conductivity"] = np.log10(cond)
        row["polymer_host"] = r["polymer_host"]
        row["anion_class"] = anion
        rows.append(row)

    out = pd.DataFrame(rows)

    # consolidate anion categories with <=3 members into "rare_other" -- a one-hot
    # column that is 1 for only 1-3 rows out of 137 is pure noise for a tree model
    # (it can never be learned from, since it's absent from 134+ of the training
    # folds in LOO); better to group them than carry 5 near-empty dummy columns
    counts = out["anion_class"].value_counts()
    rare = set(counts[counts <= 3].index)
    out["anion_class"] = out["anion_class"].apply(lambda a: "rare_other" if a in rare else a)
    print(f"Consolidated {len(rare)} rare anion categories (<=3 members each) into 'rare_other': {sorted(rare)}")

    anion_dummies = pd.get_dummies(out["anion_class"], prefix="anion")
    out = pd.concat([out, anion_dummies], axis=1)

    # median-impute missing numeric concentration fields and flag imputation,
    # since "not specified" is common and dropping those rows would waste ~30% of the data
    for col in ["salt_wt_pct", "salt_molarity_M", "salt_mol_pct"]:
        out[col + "_imputed"] = out[col].isna().astype(int)
        out[col] = out[col].fillna(out[col].median())

    out = out.drop(columns=["notes_text"])
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows x {out.shape[1]} cols -> {OUT}")
    print("Anion class distribution:\n", out["anion_class"].value_counts())
    print("\nConcentration known:", int(out["concentration_known"].sum()), "/", len(out))


if __name__ == "__main__":
    main()
