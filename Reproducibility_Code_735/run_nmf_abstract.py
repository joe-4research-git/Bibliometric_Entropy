from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)


DOMAIN_PHRASES = [
    "maximum entropy",
    "relative entropy",
    "conditional entropy",
    "entropy rate",
    "cross entropy",
    "cross-entropy",
    "shannon entropy",
    "sample entropy",
    "approximate entropy",
    "permutation entropy",
    "spectral entropy",
    "semantic entropy",
    "language entropy",
    "text entropy",
    "information entropy",
    "renyi entropy",
    "jensen-shannon divergence",
    "kullback-leibler divergence",
    "mutual information",
    "language model",
    "language models",
    "large language model",
    "large language models",
    "machine translation",
    "speech recognition",
    "natural language processing",
    "natural language generation",
    "natural language understanding",
    "authorship attribution",
    "authorship identification",
    "information density",
    "information theory",
    "word order",
    "word frequency",
    "lexical diversity",
    "linguistic complexity",
    "language acquisition",
    "language processing",
    "language variation",
    "language change",
    "language evolution",
    "code switching",
    "code-switching",
    "sign language",
    "speech synthesis",
    "sentiment analysis",
    "topic model",
    "topic modeling",
    "keyword extraction",
    "text classification",
    "part of speech",
    "part-of-speech",
]


GENERIC_ACADEMIC_WORDS = {
    "abstract",
    "analysis",
    "approach",
    "article",
    "background",
    "based",
    "conclusion",
    "conclusions",
    "data",
    "design",
    "different",
    "finding",
    "findings",
    "however",
    "implication",
    "implications",
    "limitation",
    "limitations",
    "method",
    "methodology",
    "methods",
    "model",
    "models",
    "objective",
    "objectives",
    "originality",
    "paper",
    "present",
    "propose",
    "proposed",
    "purpose",
    "research",
    "result",
    "results",
    "show",
    "shows",
    "study",
    "studies",
    "technique",
    "using",
    "used",
    "value",
    "various",
}


STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | GENERIC_ACADEMIC_WORDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run abstract-only NMF topic modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("nmf_input_484.csv"),
        help="Input CSV containing at least title and abstract columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nmf_output_abstract"),
    )
    parser.add_argument(
        "--k",
        default="auto",
        help="Integer topic count or 'auto' to evaluate min-k through max-k.",
    )
    parser.add_argument("--min-k", type=int, default=6)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--min-tokens", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--init", default="nndsvda")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--beta-loss",
        choices=("frobenius", "kullback-leibler"),
        default="frobenius",
    )
    parser.add_argument("--coherence-weight", type=float, default=0.45)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    parser.add_argument("--reconstruction-weight", type=float, default=0.20)
    parser.add_argument(
        "--selection-metric",
        choices=("composite", "mean_rank"),
        default="composite",
    )
    return parser.parse_args()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "UT": "ut",
            "DOI": "doi",
            "Title": "title",
            "Abstract": "abstract",
            "Authors": "authors",
            "Year": "year",
            "Source": "source",
            "Entropy Measure(s)": "entropy_measure",
        }
    )
    required = {"title", "abstract"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return renamed


def replace_domain_phrases(text: str) -> str:
    normalized = text.lower()
    for phrase in sorted(DOMAIN_PHRASES, key=len, reverse=True):
        replacement = phrase.replace("-", " ").replace(" ", "_")
        normalized = normalized.replace(phrase, replacement)
    return normalized


def clean_abstract(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"https?://\S+|www\.\S+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\S+@\S+", " ", value)
    value = re.sub(r"\(sic\)", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsic\b", " ", value, flags=re.IGNORECASE)
    value = replace_domain_phrases(value)
    value = re.sub(r"[^a-z0-9_]+", " ", value)
    tokens = [
        token
        for token in value.split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens)


def prepare_input(
    frame: pd.DataFrame,
    min_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["abstract"] = frame["abstract"].fillna("").astype(str)
    frame["sic_marker_count"] = frame["abstract"].str.count(
        r"\(sic\)",
        flags=re.IGNORECASE,
    )
    frame["nmf_text"] = frame["abstract"].map(clean_abstract)
    frame["analysis_text_length"] = frame["nmf_text"].str.split().str.len()

    flags = []
    for row in frame.itertuples(index=False):
        if not str(row.abstract).strip():
            flags.append("empty_abstract")
        elif row.sic_marker_count >= 10:
            flags.append("corrupted_abstract_sic")
        elif row.analysis_text_length < min_tokens:
            flags.append("abstract_too_short")
        else:
            flags.append("")
    frame["abstract_quality_flag"] = flags
    frame["eligible_for_nmf"] = frame["analysis_text_length"] >= min_tokens

    eligible = frame.loc[frame["eligible_for_nmf"]].reset_index(drop=True)
    excluded = frame.loc[~frame["eligible_for_nmf"]].copy()
    return eligible, excluded


def umass_coherence(
    topic_terms: list[list[str]],
    binary_count_matrix,
    vocabulary: dict[str, int],
) -> float:
    document_frequency = np.asarray(
        binary_count_matrix.sum(axis=0)
    ).ravel()
    values = []
    for terms in topic_terms:
        indices = [
            vocabulary[term]
            for term in terms
            if term in vocabulary
        ]
        for left_position, left in enumerate(indices):
            for right in indices[:left_position]:
                co_documents = int(
                    binary_count_matrix[:, left]
                    .multiply(binary_count_matrix[:, right])
                    .sum()
                )
                left_documents = int(document_frequency[left])
                if left_documents > 0:
                    values.append(
                        np.log((co_documents + 1.0) / left_documents)
                    )
    return float(np.mean(values)) if values else float("nan")


def topic_diversity(topic_terms: list[list[str]]) -> float:
    terms = [term for topic in topic_terms for term in topic]
    return len(set(terms)) / len(terms) if terms else 0.0


def top_topic_terms(
    model: NMF,
    feature_names: np.ndarray,
    topic_id: int,
    top_n: int,
) -> list[tuple[str, float]]:
    weights = model.components_[topic_id]
    indices = weights.argsort()[::-1][:top_n]
    return [
        (str(feature_names[index]), float(weights[index]))
        for index in indices
    ]


def evaluate_k(
    documents: list[str],
    k: int,
    args: argparse.Namespace,
):
    tfidf = TfidfVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
    )
    tfidf_matrix = tfidf.fit_transform(documents)

    model = NMF(
        n_components=k,
        init=args.init,
        random_state=args.random_state,
        max_iter=args.max_iter,
        tol=args.tol,
        beta_loss=args.beta_loss,
    )
    document_topic = model.fit_transform(tfidf_matrix)
    feature_names = tfidf.get_feature_names_out()
    topic_terms = [
        [
            term
            for term, _ in top_topic_terms(
                model,
                feature_names,
                topic_id,
                args.top_n,
            )
        ]
        for topic_id in range(k)
    ]

    count_vectorizer = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        binary=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
    )
    count_matrix = count_vectorizer.fit_transform(documents)
    coherence = umass_coherence(
        topic_terms,
        count_matrix,
        count_vectorizer.vocabulary_,
    )

    term_rows = []
    for topic_id in range(k):
        for rank, (term, weight) in enumerate(
            top_topic_terms(
                model,
                feature_names,
                topic_id,
                args.top_n,
            ),
            start=1,
        ):
            term_rows.append(
                {
                    "nmf_topic": topic_id,
                    "rank": rank,
                    "term": term,
                    "term_display": term.replace("_", " "),
                    "weight": weight,
                }
            )

    metrics = {
        "k": k,
        "reconstruction_error": float(model.reconstruction_err_),
        "umass_coherence": coherence,
        "topic_diversity": topic_diversity(topic_terms),
        "mean_max_topic_loading": float(
            document_topic.max(axis=1).mean()
        ),
        "n_iter": int(model.n_iter_),
    }
    return {
        "metrics": metrics,
        "terms": pd.DataFrame(term_rows),
        "topic_terms": topic_terms,
        "document_topic": document_topic,
        "model": model,
        "vectorizer": tfidf,
    }


def minmax(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return pd.Series(1.0, index=values.index)
    scaled = (values - minimum) / (maximum - minimum)
    return scaled if higher_is_better else 1.0 - scaled


def add_selection_scores(
    metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["coherence_score"] = minmax(
        metrics["umass_coherence"],
        higher_is_better=True,
    )
    metrics["diversity_score"] = minmax(
        metrics["topic_diversity"],
        higher_is_better=True,
    )
    metrics["reconstruction_score"] = minmax(
        metrics["reconstruction_error"],
        higher_is_better=False,
    )
    weight_sum = (
        args.coherence_weight
        + args.diversity_weight
        + args.reconstruction_weight
    )
    if weight_sum <= 0:
        raise ValueError("Selection weights must sum to a positive value.")
    metrics["selection_score"] = (
        args.coherence_weight * metrics["coherence_score"]
        + args.diversity_weight * metrics["diversity_score"]
        + args.reconstruction_weight * metrics["reconstruction_score"]
    ) / weight_sum

    metrics["coherence_rank"] = metrics["umass_coherence"].rank(
        ascending=False,
        method="min",
    )
    metrics["diversity_rank"] = metrics["topic_diversity"].rank(
        ascending=False,
        method="min",
    )
    metrics["reconstruction_rank"] = metrics["reconstruction_error"].rank(
        ascending=True,
        method="min",
    )
    metrics["mean_rank"] = metrics[
        [
            "coherence_rank",
            "diversity_rank",
            "reconstruction_rank",
        ]
    ].mean(axis=1)
    return metrics


def build_topic_outputs(
    selected: dict,
    corpus: pd.DataFrame,
    selected_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    document_topic = selected["document_topic"]
    assignments = corpus[
        [
            column
            for column in [
                "record_id",
                "ut",
                "doi",
                "title",
                "year",
                "source",
                "entropy_measure",
                "final_theme",
                "inclusion_origin",
                "analysis_text_length",
                "abstract_quality_flag",
            ]
            if column in corpus.columns
        ]
    ].copy()
    assignments["nmf_topic"] = document_topic.argmax(axis=1)
    assignments["nmf_max_topic_loading"] = document_topic.max(axis=1)
    for topic_id in range(selected_k):
        assignments[f"nmf_loading_{topic_id:02d}"] = document_topic[
            :, topic_id
        ]
    assignments = assignments.sort_values(
        ["nmf_topic", "nmf_max_topic_loading"],
        ascending=[True, False],
    )

    topic_terms = selected["topic_terms"]
    topic_labels = {
        topic_id: " | ".join(
            term.replace("_", " ")
            for term in terms[:10]
        )
        for topic_id, terms in enumerate(topic_terms)
    }
    topic_sizes = assignments.groupby("nmf_topic").size()
    summary = pd.DataFrame(
        [
            {
                "nmf_topic": topic_id,
                "documents": int(topic_sizes.get(topic_id, 0)),
                "top_terms": topic_labels[topic_id],
            }
            for topic_id in range(selected_k)
        ]
    )

    representative_rows = []
    for topic_id in range(selected_k):
        subset = assignments.loc[
            assignments["nmf_topic"].eq(topic_id)
        ].head(5)
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            representative_rows.append(
                {
                    "nmf_topic": topic_id,
                    "top_terms": topic_labels[topic_id],
                    "rank": rank,
                    "record_id": row.get("record_id", ""),
                    "ut": row.get("ut", ""),
                    "doi": row.get("doi", ""),
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "source": row.get("source", ""),
                    "nmf_max_topic_loading": row.get(
                        "nmf_max_topic_loading",
                        np.nan,
                    ),
                }
            )
    representatives = pd.DataFrame(representative_rows)
    return assignments, summary, representatives


def write_text_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.input,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame = normalize_columns(raw)
    corpus, excluded = prepare_input(frame, args.min_tokens)
    if len(corpus) < max(args.min_k, 2):
        raise ValueError(
            f"Only {len(corpus)} eligible documents remain; "
            "the model cannot be fitted."
        )

    documents = corpus["nmf_text"].tolist()
    candidate_k = (
        list(range(args.min_k, args.max_k + 1))
        if str(args.k).lower() == "auto"
        else [int(args.k)]
    )

    fitted = {}
    metric_rows = []
    for k in candidate_k:
        print(f"Fitting NMF with k={k}")
        result = evaluate_k(documents, k, args)
        fitted[k] = result
        metric_rows.append(result["metrics"])

    metrics = add_selection_scores(
        pd.DataFrame(metric_rows),
        args,
    )
    if args.selection_metric == "composite":
        selected_k = int(
            metrics.sort_values(
                ["selection_score", "mean_rank"],
                ascending=[False, True],
            ).iloc[0]["k"]
        )
    else:
        selected_k = int(
            metrics.sort_values(
                ["mean_rank", "selection_score"],
                ascending=[True, False],
            ).iloc[0]["k"]
        )

    selected = fitted[selected_k]
    assignments, summary, representatives = build_topic_outputs(
        selected,
        corpus,
        selected_k,
    )

    corpus_output_columns = [
        column
        for column in [
            "record_id",
            "ut",
            "doi",
            "title",
            "year",
            "source",
            "entropy_measure",
            "final_theme",
            "inclusion_origin",
            "abstract",
            "nmf_text",
            "analysis_text_length",
            "abstract_quality_flag",
        ]
        if column in corpus.columns
    ]

    metrics.to_csv(
        args.output_dir / "nmf_k_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected["terms"].to_csv(
        args.output_dir / "nmf_topic_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "nmf_topic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assignments.to_csv(
        args.output_dir / "nmf_document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    representatives.to_csv(
        args.output_dir / "nmf_topic_representative_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corpus[corpus_output_columns].to_csv(
        args.output_dir / "nmf_analysis_corpus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        args.output_dir / "nmf_excluded_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_text_list(
        args.output_dir / "stopwords.txt",
        STOPWORDS,
    )
    write_text_list(
        args.output_dir / "domain_phrases.txt",
        DOMAIN_PHRASES,
    )

    metadata = {
        "input_file": str(args.input),
        "input_documents": int(len(raw)),
        "eligible_documents": int(len(corpus)),
        "excluded_documents": int(len(excluded)),
        "excluded_by_reason": excluded[
            "abstract_quality_flag"
        ].value_counts().to_dict()
        if not excluded.empty
        else {},
        "text_field": "abstract only",
        "lemmatization": "none",
        "lowercase": True,
        "domain_phrases": "spaces and hyphens replaced with underscores",
        "stopwords": "scikit-learn English stopwords plus generic academic terms",
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "token_pattern": r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
        },
        "nmf": {
            "init": args.init,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "beta_loss": args.beta_loss,
        },
        "candidate_k": candidate_k,
        "selected_k": selected_k,
        "selection_metric": args.selection_metric,
        "selection_weights": {
            "coherence": args.coherence_weight,
            "diversity": args.diversity_weight,
            "reconstruction": args.reconstruction_weight,
        },
    }
    (args.output_dir / "nmf_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Eligible documents: {len(corpus)}")
    print(f"Excluded documents: {len(excluded)}")
    print(f"Selected k: {selected_k}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
