#!/usr/bin/env python
"""
WoS query and deterministic rule-based screening.

This file combines the two parts of the first analysis stage:

1. The reproducible Web of Science Core Collection query.
2. A deterministic screening script that processes the exported records,
   removes duplicate UT values, applies the exclusion hierarchy, and prepares
   the strong- and weak-language-signal review queues.

The WoS export should contain, at minimum, fields corresponding to:
UT, Title, Abstract, Author Keywords, Keywords Plus, Document Type,
WoS Categories, and Publication Year. Common WoS abbreviations such as TI,
AB, DE, ID, DT, WC, and PY are recognised.

Example
-------
python 01_wos_query_and_rule_screening.py \
    --input wos_entropy_core.csv \
    --output-dir screening_output \
    --strict

This script reproduces the initial automated screening only. The historical
strong/weak queues are retained for traceability. Dual-reviewer audits,
targeted re-screening (Paths A and B), and final adjudication are documented
in Supplementary Materials 2–4 and are not executed by this script.

`--strict` checks the expected historical counts:
11,516 unique records -> 6,501 excluded + 5,015 retained;
5,015 retained -> 547 strong-signal + 4,468 weak-signal records.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Reproducible Web of Science query
# ============================================================

SEARCH_DATE = "2026-09-17"

# Entropy-family terms (E in the manuscript).
ENTROPY_TERMS = (
    'entrop* OR "Kullback-Leibler" OR "KL divergence" '
    'OR "Jensen-Shannon"'
)

# Language and linguistics terms (L in the manuscript).
LANGUAGE_TERMS = (
    "language* OR linguistic* OR linguistics OR speech OR phonetic* "
    "OR phonolog* OR syntax OR syntactic* OR lexical* OR lexic* "
    "OR semantic* OR pragmatic* OR discourse* OR corpus OR corpora "
    "OR text* OR word* OR sentence* OR reading OR writing OR translation "
    "OR translated OR translating OR interpreting OR bilingual* "
    "OR multilingual* OR dialogue* OR spoken OR authorship OR literary "
    "OR typology OR parsing OR \"natural language processing\" "
    "OR \"language model*\" OR \"computational linguistic*\" "
    "OR \"language acquisition\" OR \"second language\" OR \"sign language\""
)

# Execute this expression in the Web of Science Core Collection.
# The two branches ensure that either entropy or the linguistic context is
# foregrounded in the title while the other component occurs in the Topic field.
WOS_QUERY = (
    f"(TI=({ENTROPY_TERMS}) AND TS=({LANGUAGE_TERMS})) "
    f"OR (TI=({LANGUAGE_TERMS}) AND TS=({ENTROPY_TERMS}))"
)

WOS_INDEXES = [
    "SCI-EXPANDED",
    "SSCI",
    "A&HCI",
    "ESCI",
    "CPCI-S",
    "CPCI-SSH",
]


# ============================================================
# 2. Input-column normalisation
# ============================================================

CANONICAL_COLUMNS = {
    "ut": [
        "ut",
        "accession_number",
        "accession_no",
    ],
    "doi": [
        "doi",
        "di",
        "digital_object_identifier",
    ],
    "title": [
        "title",
        "ti",
        "article_title",
    ],
    "abstract": [
        "abstract",
        "ab",
    ],
    "authors": [
        "authors",
        "au",
    ],
    "author_keywords": [
        "author_keywords",
        "de",
        "author_keywords_de",
    ],
    "keywords_plus": [
        "keywords_plus",
        "id",
        "keywords_plus_id",
    ],
    "source": [
        "source",
        "source_title",
        "so",
    ],
    "document_type": [
        "document_type",
        "dt",
    ],
    "year": [
        "year",
        "publication_year",
        "py",
    ],
    "wos_categories": [
        "wos_categories",
        "web_of_science_categories",
        "wc",
    ],
    "research_areas": [
        "research_areas",
        "sc",
    ],
}


def normalise_column_name(value: object) -> str:
    """Convert a WoS header to a forgiving snake-case lookup key."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common WoS short tags/full headers to stable internal names."""
    frame = frame.copy()
    normalised = {
        normalise_column_name(column): column
        for column in frame.columns
    }

    rename_map = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            source = normalised.get(alias)
            if source is not None:
                rename_map[source] = canonical
                break

    frame = frame.rename(columns=rename_map)

    required = {
        "ut",
        "title",
        "abstract",
        "author_keywords",
        "keywords_plus",
        "document_type",
        "wos_categories",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing required WoS fields after column normalisation: "
            f"{sorted(missing)}"
        )

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    return frame


def read_wos_export(path: Path) -> pd.DataFrame:
    """Read CSV, TSV/TXT, or Excel output exported from Web of Science."""
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype=str, keep_default_na=False)
    elif suffix in {".txt", ".tsv"}:
        frame = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    elif suffix == ".csv":
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    return standardise_columns(frame)


# ============================================================
# 3. Language and exclusion vocabularies
# ============================================================

# Strong signals identify a language object, linguistic level, language-use
# context, or a language-processing task.
STRONG_LANGUAGE_PATTERNS = [
    r"\blanguage(?:s)?\b",
    r"\blinguistic(?:s|ally)?\b",
    r"\bspeech\b",
    r"\bphonetic(?:s|ally)?\b",
    r"\bphonolog(?:y|ical|ically)\b",
    r"\bsyntax\b",
    r"\bsyntactic(?:ally)?\b",
    r"\blexical(?:ly)?\b",
    r"\blexicon\b",
    r"\bpragmatic(?:s|ally)?\b",
    r"\bdiscourse\b",
    r"\bcorpus\b",
    r"\bcorpora\b",
    r"\btranslat(?:e|ed|es|ing|ion|ions|or|ors)\b",
    r"\binterpret(?:ing|ation|ations|er|ers)\b",
    r"\bbilingual(?:ism|s)?\b",
    r"\bmultilingual(?:ism|s)?\b",
    r"\bdialogue(?:s)?\b",
    r"\bspoken\b",
    r"\bpronunciation\b",
    r"\butterance(?:s)?\b",
    r"\blanguage model(?:s)?\b",
    r"\bnatural language processing\b",
    r"\bnlp\b",
    r"\bmachine translation\b",
    r"\bquestion answering\b",
    r"\bsummar(?:y|ies|ization|isation|ize|ise|ized|ised|izing|ising)\b",
    r"\bsentiment analysis\b",
    r"\bauthorship\b",
    r"\breadability\b",
    r"\blexicograph(?:y|er|ers|ic)\b",
    r"\bterminolog(?:y|ies|ical)\b",
    r"\bmorpholog(?:ical|ically) analys(?:is|es|e|ed|ing)\b",
    r"\bmorpholog(?:ical|ically) tag(?:s|ging|ged)?\b",
    r"\bmorpholog(?:ical|ically) system(?:s)?\b",
    r"\bmorpholog(?:ical|ically) complex(?:ity|ities)\b",
    r"\bmorphosyntactic(?:s|ally)?\b",
    r"\bunknown word problem\b",
    r"\bmorphological disambiguation\b",
    r"\bmorphological segmentation\b",
    r"\bnatural language(?:s)?\b",
    r"\blanguage understanding\b",
    r"\blanguage generation\b",
    r"\bsecond language\b",
    r"\blanguage acquisition\b",
    r"\blanguage learning\b",
    r"\blanguage teaching\b",
    r"\bsociolinguistic(?:s|ally)?\b",
    r"\bpsycholinguistic(?:s|ally)?\b",
    r"\bneurolinguistic(?:s|ally)?\b",
    r"\bsign language\b",
]

# Moderate signals are deliberately insufficient for automatic inclusion.
MODERATE_LANGUAGE_PATTERNS = [
    r"\bsemantic(?:s|ally)?\b",
    r"\btext(?:s|ual|ually)?\b",
    r"\bword(?:s)?\b",
    r"\bsentence(?:s)?\b",
    r"\breading\b",
    r"\bwriting\b",
    r"\bacoustic(?:s|ally)?\b",
    r"\bspeaker(?:s)?\b",
    r"\bcommunicat(?:e|es|ed|ing|ion|ions)\b",
    r"\bconversation(?:s|al|ally)?\b",
    r"\bdialog(?:ue|ues|ic|ical|ically)?\b",
]

# The following vocabularies identify common non-language uses of entropy.
MATERIALS_PATTERNS = [
    r"\bhigh[- ]entropy alloy(?:s)?\b",
    r"\bmedium[- ]entropy alloy(?:s)?\b",
    r"\balloy(?:s)?\b",
    r"\bnanoparticle(?:s)?\b",
    r"\bnanomaterial(?:s)?\b",
    r"\bcatalyst(?:s|ic)?\b",
    r"\belectrocatalyst(?:s|ic)?\b",
    r"\boxide(?:s)?\b",
    r"\bmicrostructure(?:s)?\b",
    r"\bphase composition\b",
    r"\bmechanical propert(?:y|ies)\b",
    r"\bmaterials science\b",
    r"\bcrystal(?:s|line|lline|llization)?\b",
    r"\bcorrosion\b",
    r"\bsyntactic foam\b",
]

VISION_PATTERNS = [
    r"\bimage segmentation\b",
    r"\bsemantic segmentation\b",
    r"\binstance segmentation\b",
    r"\bobject detection\b",
    r"\bcomputer vision\b",
    r"\bremote sensing\b",
    r"\bmedical imag(?:e|es|ing)\b",
    r"\bradiolog(?:y|ical|ically)\b",
    r"\bimage classification\b",
    r"\bimage retrieval\b",
    r"\bimage compression\b",
    r"\bvideo(?:s)?\b",
    r"\blidar\b",
    r"\brgb(?:-d)?\b",
]

DECISION_PATTERNS = [
    r"\bfuzzy linguistic\b",
    r"\bprobabilistic linguistic\b",
    r"\bhesitant fuzzy linguistic\b",
    r"\bmulti[- ]criteria decision\b",
    r"\bmultiple criteria decision\b",
    r"\bmcdm\b",
    r"\bvikor\b",
    r"\btopsis\b",
    r"\bgroup decision[- ]making\b",
]

PHYSICS_PATTERNS = [
    r"\btopological entropy\b",
    r"\bdynamical system(?:s)?\b",
    r"\bpartial differential equation(?:s)?\b",
    r"\bpde(?:s)?\b",
    r"\bmanifold(?:s)?\b",
    r"\bturbulence\b",
    r"\bquantum\b",
    r"\bthermodynamic(?:s|al|ally)?\b",
    r"\bblack hole(?:s)?\b",
]

# A language task can rescue a physics/mathematics record from exclusion.
LANGUAGE_TASK_RESCUE_PATTERNS = [
    r"\bnatural language processing\b",
    r"\bmachine translation\b",
    r"\bquestion answering\b",
    r"\bhallucination(?:s)?\b",
    r"\bsemantic entropy\b",
    r"\btranslation\b",
    r"\btext classification\b",
    r"\btext mining\b",
    r"\btext generation\b",
    r"\btext simplification\b",
    r"\btext corpus\b",
    r"\bspeech\b",
    r"\bsentence(?:s)?\b",
    r"\bcorpus\b",
    r"\bcorpora\b",
    r"\bdiscourse\b",
    r"\bsyntax\b",
    r"\bsyntactic(?:ally)?\b",
    r"\blexical(?:ly)?\b",
    r"\breading\b",
    r"\bwriting\b",
    r"\bsentiment analysis\b",
    r"\bsummar(?:y|ies|ization|isation|ize|ise|ized|ised|izing|ising)\b",
    r"\bdialogue(?:s)?\b",
    r"\blanguage acquisition\b",
    r"\bsecond language\b",
    r"\bauthorship\b",
    r"\breadability\b",
]

# Materials records are rescued only by an explicit language-related title.
MATERIALS_LANGUAGE_RESCUE_TITLE_PATTERNS = [
    r"\btranslat(?:e|ed|es|ing|ion|ions|or|ors)\b",
    r"\bmachine translation\b",
    r"\bspeech\b",
    r"\bspoken\b",
    r"\bspeech recognition\b",
    r"\bsign language\b",
    r"\blinguistic(?:s|ally)?\b",
    r"\blanguage acquisition\b",
    r"\bsecond language\b",
    r"\blanguage learning\b",
    r"\blanguage teaching\b",
    r"\bcorpus\b",
    r"\bcorpora\b",
    r"\bsyntax\b",
    r"\bsyntactic(?:ally)?\b",
    r"\bsemantic entropy\b",
    r"\bsentiment analysis\b",
    r"\btext classification\b",
    r"\btext mining\b",
    r"\bquestion answering\b",
    r"\bhallucination(?:s)?\b",
    r"\bdiscourse\b",
    r"\breadability\b",
    r"\bauthorship\b",
]

# Used only to split the retained pool into the manuscript's strong/weak
# title-signal groups before manual review.
TITLE_CORE_LANGUAGE_PATTERNS = [
    r"\blinguistic(?:s|ally)?\b",
    r"\blanguage(?:s)?\b",
    r"\bspeech\b",
    r"\bphonetic(?:s|ally)?\b",
    r"\bphonolog(?:y|ical|ically)\b",
    r"\bsyntax\b",
    r"\bsyntactic(?:ally)?\b",
    r"\blexical(?:ly)?\b",
    r"\blexicon\b",
    r"\bsemantic(?:s|ally)?\b",
    r"\bpragmatic(?:s|ally)?\b",
    r"\bdiscourse\b",
    r"\bcorpus\b",
    r"\bcorpora\b",
    r"\btranslat(?:e|ed|es|ing|ion|ions|or|ors)\b",
    r"\binterpret(?:ing|ation|ations|er|ers)\b",
    r"\bbilingual(?:ism|s)?\b",
    r"\bmultilingual(?:ism|s)?\b",
    r"\breading\b",
    r"\bwriting\b",
    r"\bdialect(?:s|al)?\b",
]

TITLE_ENTROPY_PATTERNS = [
    r"\bentrop(?:y|ies|ic|ically)\b",
    r"\bkullback-leibler\b",
    r"\bkl divergence\b",
    r"\bjensen-shannon\b",
]

LANGUAGE_RELATED_WOS_CATEGORIES = [
    "Linguistics",
    "Language & Linguistics",
    "Computer Science, Artificial Intelligence",
    "Audiology & Speech-Language Pathology",
    "Psychology, Experimental",
]


# ============================================================
# 4. Screening functions
# ============================================================

def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def matching_patterns(patterns: list[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if has(pattern, text)]


def classify_record(row: pd.Series) -> tuple[str, str, list[str]]:
    """
    Apply the exclusion hierarchy in manuscript order.

    Return:
      decision: include | review | exclude
      reason: stable machine-readable reason code
      evidence: matching regular expressions supporting the decision
    """
    title = row["title"]
    text = " ".join(
        [
            title,
            row["abstract"],
            row["author_keywords"],
            row["keywords_plus"],
        ]
    )

    language = matching_patterns(STRONG_LANGUAGE_PATTERNS, text)
    moderate = matching_patterns(MODERATE_LANGUAGE_PATTERNS, text)
    materials = matching_patterns(MATERIALS_PATTERNS, text)
    vision = matching_patterns(VISION_PATTERNS, text)
    decision_science = matching_patterns(DECISION_PATTERNS, text)
    physics = matching_patterns(PHYSICS_PATTERNS, text)
    language_task = matching_patterns(
        LANGUAGE_TASK_RESCUE_PATTERNS,
        text,
    )
    title_language = matching_patterns(
        STRONG_LANGUAGE_PATTERNS,
        title,
    )
    materials_title_rescue = matching_patterns(
        MATERIALS_LANGUAGE_RESCUE_TITLE_PATTERNS,
        title,
    )

    # Rule 1: retracted records are always excluded.
    if "retracted publication" in row["document_type"].lower():
        return "exclude", "retracted_publication", []

    # Rule 2: fuzzy/probabilistic linguistic decision science is excluded
    # unless an explicit language-model context is present.
    if decision_science and not has(r"\blanguage model(?:s)?\b", text):
        return (
            "exclude",
            "fuzzy_linguistic_decision_science",
            decision_science,
        )

    # Rule 3: physics/mathematics records are excluded unless rescued by a
    # clearly described language task.
    if physics and not language_task:
        return "exclude", "non_language_physics_or_math", physics

    # Rule 4: materials/chemistry records require a language-related title.
    if materials and not materials_title_rescue:
        return "exclude", "materials_or_chemistry", materials

    # Rule 5: computer-vision/medical-imaging records require a title-level
    # language signal or a materials/language rescue title.
    if vision and not title_language and not materials_title_rescue:
        return (
            "exclude",
            "computer_vision_or_medical_imaging",
            vision,
        )

    if language:
        # Mixed language/materials records are retained for manual review unless
        # the text also contains an unambiguous language-technology task.
        if materials and not any(
            has(pattern, text)
            for pattern in [
                r"\blanguage model(?:s)?\b",
                r"\bllm(?:s)?\b",
                r"\bnatural language processing\b",
                r"\bmachine translation\b",
                r"\bquestion answering\b",
                r"\btext\b",
                r"\bspeech\b",
                r"\btranslation\b",
            ]
        ):
            return (
                "review",
                "language_and_materials_signal",
                language + materials,
            )
        return "include", "strong_language_signal", language

    # Moderate signals only qualify for manual review when the WoS category is
    # explicitly language-related.
    if moderate and any(
        category in row["wos_categories"]
        for category in LANGUAGE_RELATED_WOS_CATEGORIES
    ):
        return "review", "moderate_language_signal", moderate

    return "exclude", "no_language_signal", []


def add_title_signal_strength(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Split the retained 5,015 records into the manuscript's review groups.

    Strong: both an entropy-family term and a core language term occur in the
    title (n=547).

    Weak: all remaining retained records (n=4,468).
    """
    frame = frame.copy()
    frame["entropy_in_title"] = frame["title"].map(
        lambda value: bool(
            matching_patterns(TITLE_ENTROPY_PATTERNS, value)
        )
    )
    frame["language_in_title"] = frame["title"].map(
        lambda value: bool(
            matching_patterns(TITLE_CORE_LANGUAGE_PATTERNS, value)
        )
    )
    frame["signal_strength"] = (
        frame["entropy_in_title"] & frame["language_in_title"]
    ).map({True: "strong", False: "weak"})
    return frame


def deduplicate_by_ut(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate WoS accession numbers while preserving first records."""
    if frame["ut"].eq("").any():
        raise ValueError("One or more records have an empty UT value.")
    duplicates = int(frame["ut"].duplicated().sum())
    unique = frame.drop_duplicates(subset=["ut"], keep="first").copy()
    return unique, duplicates


def validate_expected_counts(summary: dict) -> None:
    """Fail if the historical reproduction counts differ from the manuscript."""
    expected = {
        "total": 11516,
        "exclude": 6501,
        "retained": 5015,
        "strong": 547,
        "weak": 4468,
    }
    actual = {
        "total": summary["unique_records"],
        "exclude": summary["decision_counts"].get("exclude", 0),
        "retained": summary["retained_records"],
        "strong": summary["signal_strength_counts"].get("strong", 0),
        "weak": summary["signal_strength_counts"].get("weak", 0),
    }
    if actual != expected:
        raise ValueError(
            "Reproduction counts do not match the manuscript. "
            f"Expected {expected}; got {actual}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the WoS rule-based screening procedure."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="WoS export in CSV, TSV/TXT, or Excel format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("screening_output"),
        help="Directory for screened files and the summary JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Validate the historical 11,516/6,501/5,015/547/4,468 counts.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_wos_export(args.input)
    unique, duplicate_count = deduplicate_by_ut(raw)

    screened_rows = []
    decision_counts = Counter()
    reason_counts = Counter()

    for _, row in unique.iterrows():
        decision, reason, evidence = classify_record(row)
        decision_counts[decision] += 1
        reason_counts[reason] += 1
        screened_rows.append(
            {
                **row.to_dict(),
                "suggested_decision": decision,
                "screen_reason": reason,
                "screen_evidence": " | ".join(evidence),
            }
        )

    screened = pd.DataFrame(screened_rows)
    retained = screened[
        screened["suggested_decision"].ne("exclude")
    ].copy()
    retained = add_title_signal_strength(retained)

    # Strong records are reviewed first, followed by weak records.
    retained["review_order"] = retained["signal_strength"].map(
        {"strong": 0, "weak": 1}
    )
    retained["year_num"] = pd.to_numeric(
        retained["year"],
        errors="coerce",
    )
    retained = retained.sort_values(
        ["review_order", "year_num", "title"],
        ascending=[True, False, True],
        na_position="last",
    )

    strong = retained[retained["signal_strength"].eq("strong")].copy()
    weak = retained[retained["signal_strength"].eq("weak")].copy()
    signal_counts = Counter(retained["signal_strength"])

    screened_path = args.output_dir / "wos_rule_screened_all.csv"
    retained_path = args.output_dir / "wos_retained_5015.csv"
    strong_path = args.output_dir / "wos_strong_signal_547.csv"
    weak_path = args.output_dir / "wos_weak_signal_4468.csv"

    screened.to_csv(screened_path, index=False, encoding="utf-8-sig")
    retained.to_csv(retained_path, index=False, encoding="utf-8-sig")
    strong.to_csv(strong_path, index=False, encoding="utf-8-sig")
    weak.to_csv(weak_path, index=False, encoding="utf-8-sig")

    summary = {
        "search_date": SEARCH_DATE,
        "wos_query": WOS_QUERY,
        "wos_indexes": WOS_INDEXES,
        "input_records": len(raw),
        "duplicates_removed_by_ut": duplicate_count,
        "unique_records": len(unique),
        "decision_counts": dict(decision_counts),
        "screen_reason_counts": dict(reason_counts.most_common()),
        "retained_records": len(retained),
        "signal_strength_counts": dict(signal_counts),
        "outputs": {
            "screened_all": str(screened_path),
            "retained": str(retained_path),
            "strong": str(strong_path),
            "weak": str(weak_path),
        },
    }

    if args.strict:
        validate_expected_counts(summary)

    summary_path = args.output_dir / "screening_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
