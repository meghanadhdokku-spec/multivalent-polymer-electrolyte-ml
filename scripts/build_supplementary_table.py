"""Build Supplementary Table S1: a clean, submission-ready list of all distinct
literature sources behind the 137-row (139 raw, incl. 2 excluded alkaline rows)
multivalent dataset, with a count of how many data points/which cations each
source contributed. Also flags low-confidence/secondary-source citations
(these were already marked per-row during data collection) so a human pass
knows exactly which of the 86 sources need extra scrutiny before submission.
"""
import re
import pandas as pd

SRC = r"C:\Polyemr Electrolyter ML project\data\raw\multivalent_literature_extract_merged_v3.csv"
OUT = r"C:\Polyemr Electrolyter ML project\manuscript\supplementary_table_S1.md"

LOW_CONFIDENCE_MARKERS = [
    "not confirmed", "unconfirmed", "low confidence", "low-confidence",
    "secondary source", "citation incomplete", "secondhand",
]


def is_low_confidence(citation, notes):
    text = f"{citation} {notes}".lower()
    return any(m in text for m in LOW_CONFIDENCE_MARKERS)


def main():
    df = pd.read_csv(SRC)
    # exclude the 2 alkaline rows, same as the modeling pipeline
    df = df[~df["salt_or_dopant"].astype(str).str.contains("KOH", case=False, na=False)]

    grouped = []
    for citation, g in df.groupby("source_citation", sort=False):
        cations = sorted(g["cation_type"].unique())
        low_conf = any(is_low_confidence(citation, n) for n in g["notes"].astype(str))
        grouped.append(
            {
                "citation": citation,
                "n_rows": len(g),
                "cations": ", ".join(cations),
                "low_confidence": low_conf,
            }
        )

    grouped.sort(key=lambda r: r["citation"].lower())

    lines = []
    lines.append("# Supplementary Table S1 — Full source list for the multivalent polymer electrolyte dataset\n")
    lines.append(
        f"All **{len(grouped)} distinct literature sources** behind the 137-row dataset used in this paper "
        "(2 alkaline/KOH-soak rows, which were excluded from modeling, are also excluded from this source "
        "list). Sources are sorted alphabetically by citation text. `n_rows` = number of data points this "
        "source contributed. Sources flagged `LOW CONFIDENCE` had incomplete citation details and/or were "
        "sourced secondarily (e.g. via a review paper's comparison table rather than independently verified "
        "against primary full text) — flagged per the project's own data-collection notes, not re-assessed "
        "here. These should be independently re-verified before this dataset is used beyond a feasibility "
        "demonstration.\n"
    )

    n_low_conf = sum(r["low_confidence"] for r in grouped)
    lines.append(f"**{n_low_conf} of {len(grouped)} sources ({n_low_conf/len(grouped)*100:.0f}%) are flagged low-confidence.**\n")

    lines.append("| # | Source | Cation(s) | N rows | Confidence |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(grouped, 1):
        flag = "LOW CONFIDENCE" if r["low_confidence"] else "OK"
        citation = r["citation"].replace("|", "\\|")
        lines.append(f"| {i} | {citation} | {r['cations']} | {r['n_rows']} | {flag} |")

    out_text = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"Wrote {len(grouped)} sources ({n_low_conf} low-confidence) -> {OUT}")


if __name__ == "__main__":
    main()
