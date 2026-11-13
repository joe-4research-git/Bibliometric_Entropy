from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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
    missing = {"title", "abstract"} - set(renamed.columns)
    if missing:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return renamed


def clean_text_for_embedding(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"https?://\S+|www\.\S+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\S+@\S+", " ", value)
    value = re.sub(r"\(sic\)", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsic\b", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def prepare_documents(frame: pd.DataFrame):
    prepared = frame.copy()
    prepared["abstract"] = prepared["abstract"].fillna("").astype(str)
    prepared["embedding_text"] = prepared["abstract"].map(
        clean_text_for_embedding
    )
    prepared["word_count"] = (
        prepared["embedding_text"].str.split().str.len()
    )
    eligible = prepared.loc[prepared["word_count"].gt(0)].reset_index(drop=True)
    excluded = prepared.loc[prepared["word_count"].eq(0)].copy()
    return eligible, excluded
