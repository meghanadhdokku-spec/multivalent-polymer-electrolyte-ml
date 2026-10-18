"""Build Supplementary Table S1: a clean, submission-ready list of all distinct
literature sources behind the 137-row (139 raw, incl. 2 excluded alkaline rows)
multivalent dataset, with a count of how many data points/which cations each
source contributed. Also flags low-confidence/secondary-source citations so a
reviewer can see at a glance which sources need extra scrutiny.
"""
import pandas as pd

SRC = r"C:\Polyemr Electrolyter ML project\data\raw\multivalent_literature_extract_merged_v3.csv"
OUT = r"C:\Polyemr Electrolyter ML project\manuscript\supplementary_table_S1.md"

# Hand-reviewed 2026-10-18 (two passes): every one of the 84 distinct sources'
# full citation and notes text was read individually and classified against
# one fixed rule -- LOW CONFIDENCE if (a) the citation's own identity
# (authors/journal/DOI) could not be confirmed, (b) the conductivity value is
# explicitly sourced secondarily (from a review paper's comparison table, or
# "cited in" another paper) rather than from the primary paper itself, or
# (c) a core identifying/quantitative detail of the system itself -- the
# conductivity value, the salt identity, or the host polymer identity, not
# just an ancillary parameter like exact wt% or measurement temperature --
# is explicitly flagged unconfirmed together with an instruction to verify
# against the full text before use. A source is NOT marked low-confidence
# merely because a secondary, non-identifying detail is unconfirmed on an
# otherwise complete, primary, fully-accessed citation (e.g. an exact molar
# ratio calculated from a stated mass recipe, or a DOI simply not published
# by the journal). Each key below is verified to match exactly one of the 84
# sources by the assertion in main() below.
#
# Third pass, 2026-10-18: using real institutional access (University of
# Tuebingen) and, for two CAPTCHA-blocked publisher pages, legitimate
# alternate sources (an institutional-repository abstract page, an open CC
# BY 4.0 Mendeley dataset page), all 11 remaining access-blocked sources
# were individually re-verified against the actual full text and removed
# from this list -- their notes/citations in the raw CSV now reflect what
# was actually confirmed, including corrections where an earlier guess
# turned out right (e.g. Pandey's salt identity) or a title/author list was
# wrong or missing entirely (e.g. Koliyoor's initials; the apclab.hcmus.edu.vn
# source's full citation, recovered from scratch).
LOW_CONFIDENCE_KEYS = [
    "A high thermally stable polyacrylonitrile",
    "Novel hybrid composites NaCF3SO3-PEO-NASICON",
    "Abdel-Samiea, B.M.",
    "Bandara, L.R.A.K.",
    "Chan, C.Y.",
    "Citation incomplete - found only via a secondary summary",
    "Citation incomplete - found via researchportal.port.ac.uk",
    "Citation incomplete: 2020 study of PEO/PVDF/NaClO4",
    "Cited as 'Diana et al.'",
    "Dalvi, A. (2020). Novel Na3Zr2Si2PO12",
    "Dalvi, A. et al. (2019). Electrical transport",
    "Deng, Y., et al. (2023)",
    "Diana, M.I., et al. (2022). Substantial ion conduction",
    "Diana, M.I., Selvasekarapandian",
    "Dong, H., et al.",
    "found via a search-engine snippet cluster (same Malaysian",
    "Found via doaj.org search snippet",
    "Genier, F.S.",
    "Huang, J., et al. (2021)",
    "Jiao, Y., et al.",
    "Karan, S., Sahu, T.B.",
    "Li, M., et al. (2015)",
    "Li, R., Zhang, L.",
    "Likely: Polymer Chemistry, 2020",
    "Lin, C., Kim, S.-H.",
    "Liu, D., et al. (2021). Self-Healing",
    "Liu, Q., et al. (2023)",
    "Lu, Y., et al. (2020)",
    "Neha, Dalvi, A.",
    "Park, B., Schaefer, J.L. (2020). Review-Polymer Electrolytes for Magnesium Batteries. Journal",
    "Park, B., Schaefer, J.L. (2020). Review-Polymer Electrolytes for Magnesium Batteries: Forging",
    "Qi, X., Ma, Q.",
    "Qiu, H., et al.",
    "Secondary citation (ref [25]",
    "Source: par.nsf.gov (NSF Public Access Repository, document ID 10126753)",
    "Source: par.nsf.gov document",
    "Source: publikationen.bibliothek.kit.edu",
    "Source: rsc.org supplementary data file",
    "Studies on sodium carboxymethyl cellulose-sodium triflate",
    "Walke, P., et al.",
    "Wang, Y., et al. (2023). Lean-water",
    "Wickramaarachchi, W.A.B.P.",
    "Wu, M., Zhang, Y.",
    "Zhang, Q., et al. (2021)",
    "Zhang, W.T., Guo, F.J.",
    # Second pass: core system-identifying detail (salt/host identity or the
    # conductivity value itself) explicitly unconfirmed with an instruction
    # to verify against full text/SI before use -- see rule (c) above.
    "Dhanush, P.C.",
    "Lin, D., Lin, Y.",
    "Sowmiya, S.",
    "Xue, Y., Quesnel",
]


def is_low_confidence(citation, notes):
    return any(key in citation for key in LOW_CONFIDENCE_KEYS)


def main():
    df = pd.read_csv(SRC)
    # exclude the 2 alkaline rows, same as the modeling pipeline
    df = df[~df["salt_or_dopant"].astype(str).str.contains("KOH", case=False, na=False)]

    citations = df["source_citation"].drop_duplicates().tolist()
    for key in LOW_CONFIDENCE_KEYS:
        hits = [c for c in citations if key in c]
        if len(hits) != 1:
            raise ValueError(f"LOW_CONFIDENCE_KEYS entry {key!r} matched {len(hits)} sources, expected 1")

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
        "source contributed. Each source was individually reviewed and flagged `LOW CONFIDENCE` if its own "
        "citation (authors, journal, or DOI) could not be confirmed, or if its conductivity value is "
        "explicitly sourced secondarily (via a review paper's comparison table rather than the primary "
        "paper itself); a source with a complete, primary citation is not flagged merely because some "
        "ancillary experimental detail (exact concentration, temperature) is unconfirmed. These should be "
        "independently re-verified before this dataset is used beyond a feasibility demonstration.\n"
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
