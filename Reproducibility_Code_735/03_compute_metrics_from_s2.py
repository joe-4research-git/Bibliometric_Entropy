#!/usr/bin/env python
"""
Compute all non-modeling indicators reported in the manuscript from
Supplementary Table S2.

This script does not fit NMF or BERTopic and does not generate figures.
It reproduces the publication, source-dispersion, theme, entropy-measure,
theme-diversity, measure-diversity, and theme-measure indicators.

Example:
  python 03_compute_metrics_from_s2.py \
      --input "Supplementary material_4624694.xlsx" \
      --output-dir metrics_from_s2 \
      --strict

Important definitions
---------------------
1. Period-based indicators include 1999-2025. The partial 2026 year is
   reported separately and excluded from period averages, period shares,
   period entropies, and HHI.
2. Figure 5 measure shares use the number of publications using a measure
   divided by all publications in that period. Multi-label shares therefore
   need not sum to 100%.
3. H_t(Measure), H_k, and measure-assignment shares use the total number of
   measure assignments. Publications may contribute multiple assignments.
4. HHI uses source_family, not raw source strings.
5. Table 6 uses the full 735-publication corpus, including 2026.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


S2_SHEET = "S2_source_theme_measure_735"

PERIODS = [
    ("1999-2003", 1999, 2003, 5),
    ("2004-2008", 2004, 2008, 5),
    ("2009-2013", 2009, 2013, 5),
    ("2014-2018", 2014, 2018, 5),
    ("2019-2023", 2019, 2023, 5),
    ("2024-2025", 2024, 2025, 2),
]

PERIOD_ORDER = [period[0] for period in PERIODS]
PERIOD_YEARS = {
    period: years
    for period, start, end, years in PERIODS
}

HHI_PERIODS = PERIOD_ORDER[:5]

THEME_ORDER = [
    "Language systems: structure, complexity and change",
    "Textual style: authorship and register",
    "Language processing: predictability and surprisal",
    "Bilingualism: language entropy and experience",
    "Entropy estimation: measures and model",
    "Language modeling: estimation and adaptation",
    "Speech and sign language: modeling and recognition",
    "Text analytics: classification, extraction and generation",
    "Cross-lingual processing: modeling, alignment and evaluation",
    "Large language models: uncertainty, reasoning and behavior",
]

MEASURE_ORDER = [
    "Shannon entropy",
    "Conditional entropy",
    "Entropy rate",
    "Cross entropy",
    "Relative entropy",
    "Rényi entropy",
    "von Neumann entropy",
    "Sequence complexity entropy",
    "Maximum entropy",
]



REQUIRED_COLUMNS = {
    "ut",
    "year",
    "source",
    "source_family",
    "entropy_measure",
    "final_theme",
}


def read_s2(
    input_path: Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Load and validate the S2 sheet."""
    suffix = input_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        available = pd.ExcelFile(input_path).sheet_names
        if sheet_name in available:
            selected_sheet = sheet_name
        else:
            candidates = [
                name
                for name in available
                if name.lower().startswith("s2")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Could not identify the S2 sheet. "
                    f"Available sheets: {available}"
                )
            selected_sheet = candidates[0]

        frame = pd.read_excel(
            input_path,
            sheet_name=selected_sheet,
            dtype=str,
            keep_default_na=False,
        )
    elif suffix == ".csv":
        frame = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    else:
        raise ValueError(
            f"Unsupported input format: {input_path.suffix}"
        )

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"S2 is missing required columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["_paper_id"] = np.arange(1, len(frame) + 1)

    for column in [
        "ut",
        "source",
        "source_family",
        "entropy_measure",
        "final_theme",
    ]:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    frame["year"] = pd.to_numeric(
        frame["year"],
        errors="coerce",
    )

    if frame["year"].isna().any() or (frame["year"] % 1 != 0).any():
        raise ValueError("S2 contains invalid or missing years.")
    frame["year"] = frame["year"].astype(int)

    if not frame["year"].between(1998, 2026).all():
        raise ValueError("S2 contains years outside 1998-2026.")

    if frame["final_theme"].eq("").any():
        raise ValueError("S2 contains empty final_theme values.")

    unknown_themes = sorted(
        set(frame["final_theme"]) - set(THEME_ORDER)
    )
    if unknown_themes:
        raise ValueError(
            f"Unknown final themes in S2: {unknown_themes}"
        )

    return frame


def assign_period(year: int) -> str | float:
    for label, start, end, _ in PERIODS:
        if start <= year <= end:
            return label
    return np.nan


def split_measure_families(value: object) -> list[str]:
    """Split semicolon-delimited measure labels and remove duplicates."""
    labels = [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]
    aliases = {label.casefold(): label for label in MEASURE_ORDER}
    aliases.update(conditional="Conditional entropy", relative="Relative entropy")
    unique_labels = list(dict.fromkeys(aliases.get(label.casefold(), label) for label in labels))
    unknown = sorted(set(unique_labels) - set(MEASURE_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown entropy-measure families in S2: {unknown}"
        )
    return unique_labels


def shannon_entropy_bits(probabilities: pd.Series) -> float:
    """Calculate base-2 Shannon entropy with 0 log(0) defined as 0."""
    values = (
        probabilities.astype(float)
        .loc[probabilities.astype(float).gt(0)]
    )
    if values.empty:
        return float("nan")
    return float(
        -(values * np.log2(values)).sum()
    )


def reorder_index(
    frame: pd.DataFrame,
    order: list[str],
) -> pd.DataFrame:
    present = [
        item
        for item in order
        if item in frame.index
    ]
    remaining = [
        item
        for item in frame.index
        if item not in present
    ]
    return frame.reindex(present + remaining)


def compute_metrics(
    papers: pd.DataFrame,
) -> dict[str, pd.DataFrame | dict]:
    """Compute every metric derivable from S2."""
    papers = papers.copy()
    papers["period"] = papers["year"].map(assign_period)
    papers["measure_list"] = papers["entropy_measure"].map(
        split_measure_families
    )
    papers["measure_count"] = papers["measure_list"].map(len)
    if papers["source_family"].str.casefold().isin({"", "nan", "none"}).any():
        raise ValueError("S2 contains missing source families.")
    papers["source_group"] = papers["source_family"]

    if papers["measure_count"].eq(0).any():
        raise ValueError(
            "At least one S2 publication has no measure assignment."
        )

    period_papers = papers[
        papers["year"].between(1999, 2025)
    ].copy()
    partial_2026 = papers[
        papers["year"].eq(2026)
    ].copy()

    if period_papers["period"].isna().any():
        raise ValueError(
            "Some 1999-2025 records could not be assigned to a period."
        )

    # --------------------------------------------------------
    # 1. Annual and period publication scale
    # --------------------------------------------------------
    annual_counts = (
        papers.groupby("year")
        .size()
        .rename("publications")
        .reset_index()
        .sort_values("year")
    )
    annual_counts["share_of_all_publications_pct"] = (
        annual_counts["publications"]
        / len(papers)
        * 100
    )

    period_totals = (
        period_papers.groupby("period")
        .size()
        .reindex(PERIOD_ORDER)
        .fillna(0)
        .astype(int)
    )

    period_summary = pd.DataFrame(
        {
            "period": PERIOD_ORDER,
            "years": [
                PERIOD_YEARS[period]
                for period in PERIOD_ORDER
            ],
            "publications": [
                int(period_totals[period])
                for period in PERIOD_ORDER
            ],
        }
    )
    period_summary["publications_per_year"] = (
        period_summary["publications"]
        / period_summary["years"]
    )
    period_summary["share_of_1999_2025_pct"] = (
        period_summary["publications"]
        / len(period_papers)
        * 100
    )

    partial_year_summary = pd.DataFrame(
        [
            {
                "year": 2026,
                "publications": int(len(partial_2026)),
                "partial_year": True,
                "included_in_period_metrics": False,
            }
        ]
    )

    # --------------------------------------------------------
    # 2. Source distribution and HHI
    # --------------------------------------------------------
    source_counts = (
        papers.groupby("source_group")
        .size()
        .rename("publications")
        .reset_index()
        .sort_values(
            ["publications", "source_group"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    source_counts["rank"] = np.arange(
        1,
        len(source_counts) + 1,
    )
    source_counts["share_pct"] = (
        source_counts["publications"]
        / len(papers)
        * 100
    )
    source_counts["cumulative_share_pct"] = (
        source_counts["share_pct"].cumsum()
    )

    source_distribution_summary = pd.DataFrame(
        [
            {
                "n_publications": len(papers),
                "n_source_groups": int(
                    papers["source_group"].nunique()
                ),
                "top5_share_pct": float(
                    source_counts.head(5)["share_pct"].sum()
                ),
                "top10_share_pct": float(
                    source_counts.head(10)["share_pct"].sum()
                ),
                "top20_share_pct": float(
                    source_counts.head(20)["share_pct"].sum()
                ),
                "long_tail_share_pct": float(
                    source_counts.iloc[20:]["share_pct"].sum()
                ),
                "long_tail_source_families": int(
                    max(0, len(source_counts) - 20)
                ),
            }
        ]
    )

    hhi_rows = []
    period_source_share_rows = []
    for period in HHI_PERIODS:
        subset = period_papers[
            period_papers["period"].eq(period)
        ]
        counts = subset["source_family"].value_counts()
        n = len(subset)
        shares = counts / n if n else counts.astype(float)
        hhi_rows.append(
            {
                "period": period,
                "publications": int(len(subset)),
                "unique_source_families": int(counts.size),
                "HHI": float((shares**2).sum()) if n else np.nan,
                "Adjusted_HHI": (
                    int((counts * (counts - 1)).sum()) / (n * (n - 1))
                    if n > 1 else np.nan
                ),
            }
        )
        for source_family, count in counts.items():
            period_source_share_rows.append(
                {
                    "period": period,
                    "source_family": source_family,
                    "publications": int(count),
                    "source_share_in_period": float(shares[source_family]),
                }
            )

    hhi = pd.DataFrame(hhi_rows)
    period_source_shares = pd.DataFrame(
        period_source_share_rows
    )

    # --------------------------------------------------------
    # 3. Research-theme period counts, rates, shares, entropy
    # --------------------------------------------------------
    theme_counts = (
        period_papers.groupby(["final_theme", "period"])
        .size()
        .rename("publications")
        .unstack(fill_value=0)
        .reindex(
            index=THEME_ORDER,
            columns=PERIOD_ORDER,
            fill_value=0,
        )
        .astype(int)
    )
    theme_count_per_year = theme_counts.div(
        pd.Series(PERIOD_YEARS),
        axis=1,
    )
    theme_share_pct = theme_counts.div(
        period_totals,
        axis=1,
    ) * 100

    theme_period_long = []
    for theme in THEME_ORDER:
        for period in PERIOD_ORDER:
            theme_period_long.append(
                {
                    "period": period,
                    "final_theme": theme,
                    "publications": int(
                        theme_counts.loc[theme, period]
                    ),
                    "publications_per_year": float(
                        theme_count_per_year.loc[theme, period]
                    ),
                    "within_period_share_pct": float(
                        theme_share_pct.loc[theme, period]
                    ),
                }
            )
    theme_period_long = pd.DataFrame(theme_period_long)

    theme_entropy = pd.DataFrame(
        [
            {
                "period": period,
                "theme_entropy_bits": shannon_entropy_bits(
                    theme_counts[period] / theme_counts[period].sum()
                ),
            }
            for period in PERIOD_ORDER
        ]
    )

    # --------------------------------------------------------
    # 4. Measure period publication rates and shares
    # --------------------------------------------------------
    measure_assignments = (
        papers[["_paper_id", "period", "measure_list"]]
        .explode("measure_list")
        .dropna(subset=["measure_list"])
        .rename(columns={"measure_list": "measure"})
        .copy()
    )
    measure_assignments = measure_assignments[
        measure_assignments["period"].notna()
    ]

    measure_paper_counts = (
        measure_assignments.groupby(["measure", "period"])
        .size()
        .rename("publications")
        .unstack(fill_value=0)
        .reindex(
            index=MEASURE_ORDER,
            columns=PERIOD_ORDER,
            fill_value=0,
        )
        .astype(int)
    )
    measure_count_per_year = measure_paper_counts.div(
        pd.Series(PERIOD_YEARS),
        axis=1,
    )
    # Denominator is all publications in the period, not all assignments.
    measure_publication_share_pct = measure_paper_counts.div(
        period_totals,
        axis=1,
    ) * 100

    assignment_totals = (
        measure_assignments.groupby("period")
        .size()
        .reindex(PERIOD_ORDER)
        .fillna(0)
        .astype(int)
    )
    measure_assignment_share = measure_paper_counts.div(
        assignment_totals,
        axis=1,
    )

    measure_period_long = []
    for measure in MEASURE_ORDER:
        for period in PERIOD_ORDER:
            measure_period_long.append(
                {
                    "period": period,
                    "entropy_measure": measure,
                    "publications_using_measure": int(
                        measure_paper_counts.loc[measure, period]
                    ),
                    "publications_per_year": float(
                        measure_count_per_year.loc[measure, period]
                    ),
                    "share_of_publications_pct": float(
                        measure_publication_share_pct.loc[
                            measure, period
                        ]
                    ),
                    "measure_assignments": int(
                        measure_paper_counts.loc[measure, period]
                    ),
                    "share_of_assignments": float(
                        measure_assignment_share.loc[
                            measure, period
                        ]
                    ),
                }
            )
    measure_period_long = pd.DataFrame(measure_period_long)

    measure_entropy = pd.DataFrame(
        [
            {
                "period": period,
                "measure_entropy_bits": shannon_entropy_bits(
                    measure_assignment_share[period]
                ),
            }
            for period in PERIOD_ORDER
        ]
    )

    temporal_entropy = theme_entropy.merge(
        measure_entropy,
        on="period",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 5. Theme x measure counts and publication-level shares
    # --------------------------------------------------------
    theme_measure_all = (
        papers[["_paper_id", "final_theme", "measure_list"]]
        .explode("measure_list")
        .dropna(subset=["measure_list"])
        .rename(columns={"measure_list": "measure"})
    )

    theme_measure_counts = (
        theme_measure_all.groupby(["final_theme", "measure"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=THEME_ORDER,
            columns=MEASURE_ORDER,
            fill_value=0,
        )
        .astype(int)
    )

    theme_paper_counts = (
        papers.groupby("final_theme")
        .size()
        .reindex(THEME_ORDER)
        .fillna(0)
        .astype(int)
    )
    theme_measure_publication_pct = (
        theme_measure_counts.div(
            theme_paper_counts,
            axis=0,
        )
        * 100
    )

    # --------------------------------------------------------
    # 6. Table 6 indicators
    # --------------------------------------------------------
    theme_assignment_counts = (
        theme_measure_counts.sum(axis=1)
        .reindex(THEME_ORDER)
        .astype(int)
    )

    table6_rows = []
    for theme in THEME_ORDER:
        distribution = theme_measure_counts.loc[theme].astype(float)
        probabilities = distribution / distribution.sum()
        assignment_count = int(theme_assignment_counts[theme])
        paper_count = int(theme_paper_counts[theme])
        mean_measures = assignment_count / paper_count
        theme_papers = papers[
            papers["final_theme"].eq(theme)
        ]
        multi_count = int(
            theme_papers["measure_count"].ge(2).sum()
        )
        multi_pct = multi_count / paper_count * 100
        hk = shannon_entropy_bits(probabilities)
        if abs(hk) < 1e-12:
            hk = 0.0

        # Descriptive conventions for Table 6; these are not validated class boundaries.
        if mean_measures >= 1.50 and multi_pct >= 40.0:
            profile = "within-study combination"
        elif hk >= 2.0:
            profile = "cross-study pluralism"
        else:
            profile = "measure-specific use"

        table6_rows.append(
            {
                "final_theme": theme,
                "papers": paper_count,
                "measure_assignments": assignment_count,
                "H_k_bits": hk,
                "mean_measures_per_paper": mean_measures,
                "multi_measure_papers": multi_count,
                "multi_measure_papers_pct": multi_pct,
                "profile": profile,
            }
        )
    table6 = pd.DataFrame(table6_rows)

    # --------------------------------------------------------
    # 7. Overall distributions
    # --------------------------------------------------------
    all_theme_counts = theme_paper_counts.rename(
        "publications"
    ).reset_index()
    all_theme_counts["share_pct"] = (
        all_theme_counts["publications"]
        / len(papers)
        * 100
    )

    all_measure_publication_counts = (
        theme_measure_counts.sum(axis=0)
        .rename("publications")
        .rename_axis("entropy_measure")
        .reset_index()
    )
    all_measure_publication_counts["share_pct"] = (
        all_measure_publication_counts["publications"]
        / len(papers)
        * 100
    )
    all_measure_assignment_counts = (
        all_measure_publication_counts.rename(
            columns={"publications": "measure_assignments"}
        )
    )
    all_measure_assignment_counts["assignment_share_pct"] = (
        all_measure_assignment_counts["measure_assignments"]
        / all_measure_assignment_counts[
            "measure_assignments"
        ].sum()
        * 100
    )

    overall_summary = pd.DataFrame(
        [
            {
                "publications": len(papers),
                "publications_2026": len(partial_2026),
                "measure_assignments": int(
                    papers["measure_count"].sum()
                ),
                "papers_with_multiple_measures": int(
                    papers["measure_count"].ge(2).sum()
                ),
                "multi_measure_papers_pct": float(
                    papers["measure_count"].ge(2).mean()
                    * 100
                ),
                "themes": int(papers["final_theme"].nunique()),
                "entropy_measure_families": int(
                    theme_measure_counts.shape[1]
                ),
                "source_families": int(
                    papers["source_family"].nunique()
                ),
                "source_groups": int(
                    papers["source_group"].nunique()
                ),
            }
        ]
    )

    manifest = {
        "input_publications": int(len(papers)),
        "period_publications_1999_2025": int(len(period_papers)),
        "pre_1999_publications": int(papers["year"].lt(1999).sum()),
        "partial_2026_publications": int(len(partial_2026)),
        "measure_assignments": int(
            papers["measure_count"].sum()
        ),
        "periods": [
            {
                "label": label,
                "start": start,
                "end": end,
                "years": years,
            }
            for label, start, end, years in PERIODS
        ],
        "hhi_periods": HHI_PERIODS,
        "definitions": {
            "HHI": "sum of squared within-period source_family shares",
            "Adjusted_HHI": "proportion of distinct within-period publication pairs sharing a source family (Simpson, 1949)",
            "source_grouping": "HHI and source distribution use the same finalized source_family values from S2.",
            "theme_entropy": (
                "-sum(theme publication share * log2(theme publication "
                "share)) within each period"
            ),
            "measure_entropy": (
                "-sum(measure-assignment share * log2(measure-assignment "
                "share)) within each period"
            ),
            "H_k": (
                "-sum(measure-assignment share * log2(measure-assignment "
                "share)) within each theme"
            ),
            "mean_measures_per_paper": (
                "measure assignments / publications in the theme"
            ),
            "multi_measure_pct": (
                "papers with >=2 unique measure families / papers in the theme"
            ),
            "profile_rules": {
                "interpretation": "Descriptive conventions, not empirically validated classes.",
                "within-study combination": (
                    "mean measures per paper >= 1.50 and "
                    "multi-measure share >= 40%"
                ),
                "cross-study pluralism": (
                    "not within-study combination and H_k >= 2.0 bits"
                ),
                "measure-specific use": "all remaining themes",
            },
        },
    }

    return {
        "annual_publication_counts": annual_counts,
        "period_publication_summary": period_summary,
        "partial_2026_summary": partial_year_summary,
        "source_distribution": source_counts,
        "source_distribution_summary": source_distribution_summary,
        "period_source_shares": period_source_shares,
        "period_hhi": hhi,
        "theme_period_counts": (
            theme_counts.reset_index()
        ),
        "theme_period_count_per_year": (
            theme_count_per_year.reset_index()
        ),
        "theme_period_share_pct": (
            theme_share_pct.reset_index()
        ),
        "theme_period_long": theme_period_long,
        "measure_period_publication_counts": (
            measure_paper_counts.reset_index()
        ),
        "measure_period_count_per_year": (
            measure_count_per_year.reset_index()
        ),
        "measure_period_publication_share_pct": (
            measure_publication_share_pct.reset_index()
        ),
        "measure_period_assignment_share": (
            measure_assignment_share.reset_index()
        ),
        "measure_period_long": measure_period_long,
        "temporal_entropy": temporal_entropy,
        "theme_measure_counts": (
            theme_measure_counts.reset_index()
        ),
        "theme_measure_publication_pct": (
            theme_measure_publication_pct.reset_index()
        ),
        "table6_theme_measure_indicators": table6,
        "overall_theme_distribution": all_theme_counts,
        "overall_measure_publication_distribution": (
            all_measure_publication_counts
        ),
        "overall_measure_assignment_distribution": (
            all_measure_assignment_counts
        ),
        "overall_summary": overall_summary,
        "manifest": manifest,
    }


def validate_expected(
    metrics: dict[str, pd.DataFrame | dict],
) -> None:
    """Validate the headline values reported in the manuscript."""
    overall = metrics["overall_summary"].iloc[0]
    expected = {
        "publications": 735,
        "publications_2026": 43,
        "measure_assignments": 950,
        "themes": 10,
        "entropy_measure_families": 9,
        "source_families": 298,
        "source_groups": 298,
    }
    actual = {
        key: int(overall[key])
        for key in expected
    }
    if actual != expected:
        raise ValueError(
            f"Expected manuscript counts {expected}; got {actual}."
        )

    hhi = metrics["period_hhi"]
    expected_hhi = {
        "1999-2003": 0.080,
        "2019-2023": 0.024,
    }
    adjusted = {row.period: round(float(row.Adjusted_HHI), 3) for row in hhi.itertuples(index=False)}
    if adjusted.get("1999-2003") != 0.062 or adjusted.get("2019-2023") != 0.019:
        raise ValueError(f"Adjusted HHI does not match the manuscript: {adjusted}")
    rounded_hhi = {
        row.period: round(float(row.HHI), 3)
        for row in hhi.itertuples(index=False)
    }
    for period, expected_value in expected_hhi.items():
        if rounded_hhi[period] != expected_value:
            raise ValueError(
                f"Expected HHI {period}={expected_value}; "
                f"got {rounded_hhi[period]}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all non-modeling manuscript indicators "
            "from Supplementary Table S2."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Supplementary workbook containing S2 or an S2 CSV file.",
    )
    parser.add_argument(
        "--sheet",
        default=S2_SHEET,
        help="S2 worksheet name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics_from_s2"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Validate manuscript headline counts and HHI values.",
    )
    args = parser.parse_args()

    papers = read_s2(args.input, args.sheet)
    metrics = compute_metrics(papers)

    if args.strict:
        validate_expected(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, value in metrics.items():
        if name == "manifest":
            continue
        value.to_csv(
            args.output_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    (
        args.output_dir / "metrics_manifest.json"
    ).write_text(
        json.dumps(
            metrics["manifest"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics["overall_summary"].iloc[0].to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
