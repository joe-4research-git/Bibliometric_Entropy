from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from text_preprocessing import (
    STOPWORDS,
    normalize_columns,
    prepare_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run abstract-only BERTopic modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("nmf_input_484.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bertopic_output_abstract"),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "SentenceTransformer model. For a scientific-abstract model, "
            "try allenai/specter2_base."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument(
        "--cluster-selection-method",
        choices=("leaf", "eom"),
        default="leaf",
    )
    parser.add_argument(
        "--nr-topics",
        default="none",
        help="'none', 'auto', or an integer.",
    )
    parser.add_argument("--top-n-words", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--outlier-strategy",
        choices=("none", "probabilities", "embeddings", "c-tf-idf"),
        default="probabilities",
    )
    parser.add_argument("--outlier-threshold", type=float, default=0.0)
    return parser.parse_args()


def parse_nr_topics(value: str):
    normalized = str(value).strip().lower()
    if normalized in {"none", ""}:
        return None
    if normalized == "auto":
        return "auto"
    return int(normalized)


def build_vectorizer(args: argparse.Namespace) -> CountVectorizer:
    return CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
    )


def load_embedding_model(args: argparse.Namespace) -> SentenceTransformer:
    kwargs = {
        "device": args.device,
    }
    if args.local_files_only:
        kwargs["local_files_only"] = True
    try:
        return SentenceTransformer(args.embedding_model, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        return SentenceTransformer(args.embedding_model, **kwargs)


def topic_term_table(topic_model: BERTopic) -> pd.DataFrame:
    rows = []
    for topic_id in sorted(topic_model.get_topics()):
        if topic_id == -1:
            continue
        for rank, (term, weight) in enumerate(
            topic_model.get_topic(topic_id),
            start=1,
        ):
            rows.append(
                {
                    "bertopic_topic": topic_id,
                    "rank": rank,
                    "term": term,
                    "weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def representative_table(topic_model: BERTopic) -> pd.DataFrame:
    rows = []
    for topic_id in sorted(topic_model.get_topics()):
        if topic_id == -1:
            continue
        representatives = topic_model.get_representative_docs(topic_id) or []
        for rank, document in enumerate(representatives, start=1):
            rows.append(
                {
                    "bertopic_topic": topic_id,
                    "rank": rank,
                    "representative_document": document,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.input,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame = normalize_columns(raw)
    corpus, excluded = prepare_documents(frame)
    documents = corpus["embedding_text"].tolist()

    embedding_model = load_embedding_model(args)
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    umap_model = UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.umap_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        prediction_data=True,
    )
    vectorizer_model = build_vectorizer(args)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=parse_nr_topics(args.nr_topics),
        top_n_words=args.top_n_words,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probabilities = topic_model.fit_transform(
        documents,
        embeddings=embeddings,
    )
    raw_topics = np.asarray(topics, dtype=int)
    reduced_topics = raw_topics.copy()

    if args.outlier_strategy != "none":
        reduce_kwargs = {
            "documents": documents,
            "topics": reduced_topics.tolist(),
            "strategy": args.outlier_strategy,
            "threshold": args.outlier_threshold,
        }
        if probabilities is not None:
            reduce_kwargs["probabilities"] = probabilities
        reduced_topics = np.asarray(
            topic_model.reduce_outliers(**reduce_kwargs),
            dtype=int,
        )
        topic_model.update_topics(
            documents,
            topics=reduced_topics.tolist(),
            vectorizer_model=vectorizer_model,
        )

    topic_info = topic_model.get_topic_info()
    terms = topic_term_table(topic_model)
    representatives = representative_table(topic_model)

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
                "word_count",
            ]
            if column in corpus.columns
        ]
    ].copy()
    assignments["bertopic_raw_topic"] = raw_topics
    assignments["bertopic_topic"] = reduced_topics
    raw_outlier_mask = raw_topics == -1
    assignments["bertopic_raw_is_outlier"] = raw_outlier_mask
    assignments["bertopic_is_outlier"] = reduced_topics == -1
    assignments["bertopic_max_probability"] = (
        np.asarray(probabilities).max(axis=1)
        if probabilities is not None
        else np.nan
    )

    raw_outlier_assignments = assignments.loc[
        assignments["bertopic_raw_is_outlier"]
    ].copy()
    raw_outlier_assignments.sort_values(
        "bertopic_max_probability",
        ascending=False,
    ).to_csv(
        args.output_dir / "bertopic_raw_outlier_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outlier_probability_summary = {}
    if probabilities is not None and not raw_outlier_assignments.empty:
        raw_outlier_probabilities = pd.to_numeric(
            raw_outlier_assignments["bertopic_max_probability"],
            errors="coerce",
        ).dropna()
        summary_rows = [
            ("raw_outliers", int(raw_outlier_mask.sum())),
            (
                "raw_outlier_share",
                float(raw_outlier_mask.mean()),
            ),
            ("final_outliers", int((reduced_topics == -1).sum())),
            ("probability_min", float(raw_outlier_probabilities.min())),
            ("probability_q25", float(raw_outlier_probabilities.quantile(0.25))),
            ("probability_median", float(raw_outlier_probabilities.median())),
            ("probability_q75", float(raw_outlier_probabilities.quantile(0.75))),
            ("probability_max", float(raw_outlier_probabilities.max())),
            (
                "probability_below_0_2",
                int((raw_outlier_probabilities < 0.2).sum()),
            ),
            (
                "probability_below_0_3",
                int((raw_outlier_probabilities < 0.3).sum()),
            ),
            (
                "probability_below_0_5",
                int((raw_outlier_probabilities < 0.5).sum()),
            ),
            (
                "probability_below_0_7",
                int((raw_outlier_probabilities < 0.7).sum()),
            ),
            (
                "probability_at_least_0_5",
                int((raw_outlier_probabilities >= 0.5).sum()),
            ),
            (
                "probability_at_least_0_7",
                int((raw_outlier_probabilities >= 0.7).sum()),
            ),
        ]
        pd.DataFrame(
            summary_rows,
            columns=["metric", "value"],
        ).to_csv(
            args.output_dir / "bertopic_outlier_probability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outlier_probability_summary = dict(summary_rows)

    assignments.sort_values(
        ["bertopic_topic", "bertopic_max_probability"],
        ascending=[True, False],
    ).to_csv(
        args.output_dir / "bertopic_document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_info.to_csv(
        args.output_dir / "bertopic_topic_info.csv",
        index=False,
        encoding="utf-8-sig",
    )
    terms.to_csv(
        args.output_dir / "bertopic_topic_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    representatives.to_csv(
        args.output_dir / "bertopic_representative_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assignments.loc[
        assignments["bertopic_topic"].eq(-1)
    ].to_csv(
        args.output_dir / "bertopic_outliers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corpus.to_csv(
        args.output_dir / "bertopic_analysis_corpus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        args.output_dir / "bertopic_excluded_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(args.output_dir / "bertopic_embeddings.npy", embeddings)
    if probabilities is not None:
        np.save(
            args.output_dir / "bertopic_probability_matrix.npy",
            np.asarray(probabilities),
        )

    metadata = {
        "input_file": str(args.input),
        "input_documents": int(len(raw)),
        "eligible_documents": int(len(corpus)),
        "excluded_no_abstract": int(len(excluded)),
        "text_field": "abstract only",
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(embeddings.shape[1]),
        "umap": {
            "n_neighbors": args.umap_neighbors,
            "n_components": args.umap_components,
            "min_dist": args.umap_min_dist,
            "metric": "cosine",
            "random_state": args.random_state,
        },
        "hdbscan": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "metric": "euclidean",
            "cluster_selection_method": args.cluster_selection_method,
        },
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.90,
            "stopwords": "scikit-learn English stopwords plus generic academic terms",
            "token_pattern": r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
        },
        "nr_topics": str(args.nr_topics),
        "top_n_words": args.top_n_words,
        "outlier_strategy": args.outlier_strategy,
        "outlier_threshold": args.outlier_threshold,
        "raw_outliers": int((raw_topics == -1).sum()),
        "raw_outlier_share": float((raw_topics == -1).mean()),
        "final_outliers": int((reduced_topics == -1).sum()),
        "raw_topic_count": int(len(set(raw_topics) - {-1})),
        "final_topic_count": int(len(set(reduced_topics) - {-1})),
        "raw_outlier_probability_summary": outlier_probability_summary,
    }
    (args.output_dir / "bertopic_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
