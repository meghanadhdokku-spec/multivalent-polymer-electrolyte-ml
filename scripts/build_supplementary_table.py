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
# BY 4.0 Mendeley dataset page), 11 access-blocked sources were individually
# re-verified against the actual full text and removed from this list --
# their notes/citations in the raw CSV now reflect what was actually
# confirmed, including corrections where an earlier guess turned out right
# (e.g. Pandey's salt identity) or a title/author list was wrong or missing
# entirely (e.g. Koliyoor's initials; the apclab.hcmus.edu.vn source's full
# citation, recovered from scratch).
#
# Fourth pass, 2026-10-18: the one remaining source with no traceable author
# (the NITK Surathkal repository record, dead link) was resolved via
# CrossRef's own API and the publisher's abstract page -- full author list
# (Singh, Janakiraman, Khalifa, Anandhan, Ghosh, Venimadhav, Biswas),
# journal, volume/pages, and DOI all confirmed, along with the exact salt
# identity (Mg(ClO4)2) and conductivity value.
#
# Fifth pass, 2026-10-18: tracked down every remaining low-confidence source
# using real institutional access, same as the fourth pass. Checked CrossRef,
# Semantic Scholar, and open-access full texts across all 48 remaining
# flagged sources, plus a direct read of the Park & Schaefer review's own
# published tables (its 33-row bucket) and reference list. This surfaced
# real corrections, not just citation gaps:
#   - Two rows (Huang et al.) were actually two different papers conflated
#     under one reference by a secondary review; split and re-attributed.
#   - One citation (Lin, Kim et al., "Matter") could not be found to exist
#     anywhere and was replaced with the real source (Chen et al., Adv.
#     Mater. 2022) -- the data values were already correct, only the
#     citation was wrong.
#   - One row (Jiao et al.) was a review-table duplication artifact (same
#     conductivity value attached to two different, only-one-real salt
#     pairings) -- the fabricated duplicate was dropped.
#   - One source (Qiu et al., a TiO2-nucleated small-molecule eutectic with
#     no polymer backbone) was dropped as out of scope for a polymer
#     electrolyte dataset, the same reasoning as the alkaline-row exclusion.
#   - Several author-initial errors were fixed (e.g. "Li, M." -> "Li, J.",
#     "Zhang, Q." -> "Zhang, X.", Dalvi's co-author order).
#   - One source (Wu et al., 72 mS/cm) had been flagged as a suspicious
#     outlier; independently confirmed genuine and the flag was lifted.
#   - One source (Dong et al.) has a confirmed host/capacity mismatch against
#     the primary paper that could not be resolved -- kept flagged rather
#     than guessed at.
#   - One row (Wang et al., EMIMBF4/V6O13) does not actually appear anywhere
#     in the cited primary paper's full text -- kept flagged as an
#     unresolved citation error, not force-matched to a replacement.
# Sixth pass, 2026-10-18: resolved the Park & Schaefer (2020) review's entire
# 33-row bucket's underlying primary citations by reading the review's own
# numbered reference list directly (IOPscience, open access -- not guessed
# via topical matching against an unordered Semantic Scholar dump). This also
# caught and corrected a wrong guess from an earlier pass (Jeong et al. 2006
# had been mis-attributed to ref [90]; it is actually ref [87]). These rows
# stay LOW CONFIDENCE per rule (b) -- the conductivity value is still sourced
# from the review's own table, not independently re-read from each of the
# ~26 individual primary papers' own text -- but every citation in the
# bucket is now a real, confirmed reference rather than a guess or an
# unresolved gap.
#
# Rows are only moved to OK here when a specific conductivity value has
# independent corroboration beyond a single review's own comparison table
# (a primary abstract, an institutional repository, or multiple agreeing
# secondary sources) -- a review-table-only match stays flagged per rule (b)
# even when the citation itself is now fully confirmed and correct.
LOW_CONFIDENCE_KEYS = [
    "Abdel-Samiea, B.M.",
    "Bandara, L.R.A.K.",
    "Wickramaarachchi, W.A.B.P.",
    "found via a search-engine snippet cluster (same Malaysian",
    "Found via doaj.org search snippet",
    "Chan, C.Y., Wang, Z., Li, Y.",
    "Citation incomplete: 2020 study of PEO/PVDF/NaClO4",
    "Source: publikationen.bibliothek.kit.edu",
    "Source: par.nsf.gov document; full citation could not be independently confirmed",
    "Wan, F., Hu, K., Liu, R.",
    "Dong, H., Li, J., Zhao, S.",
    "UNRESOLVED (attributed by a review to Wang, Y., et al.",
    "Liu, D., Tang, Z., Luo, L.",
    "Huang, J., et al. (2021). Ultrastable Zinc Anodes",
    "Deng, Y., Wu, Y., Wang, L.",
    "Liu, Q., Yu, Z., Zhuang, Q.",
    "Lu, Y., Wen, Y., Huang, F.",
    "Citation incomplete - found only via a secondary summary",
    "Park, B., Schaefer, J.L. (2020). Review-Polymer Electrolytes for Magnesium Batteries: Forging",
    "Park, B., Schaefer, J.L. (2020). Review-Polymer Electrolytes for Magnesium Batteries. Journal",
    "Qi, X., Ma, Q., Liu, L.",
    "Genier, F.S., Pathreeker",
    "Li, J., Zhu, H., Wang, X.",
    "Zhang, X., Tang, S., Li, X.",
    "Zhang, W.T., Guo, F.J.",
    # Second pass: core system-identifying detail (salt/host identity or the
    # conductivity value itself) explicitly unconfirmed with an instruction
    # to verify against full text/SI before use -- see rule (c) above.
    "Lin, D., Lin, Y.",
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
