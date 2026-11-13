"""Run the published 735-abstract topic analysis and compare outlier treatments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from compare_conditions import main as compare

ROOT = Path(__file__).resolve().parent


def check_reassignment(output):
    output = Path(output)
    run = output / "bertopic_selective"
    corpus = pd.read_csv(run / "bertopic_analysis_corpus.csv").set_index("record_id", verify_integrity=True)
    docs = pd.read_csv(output / "document_condition_comparison.csv").set_index("record_id", verify_integrity=True)
    docs = docs.loc[corpus.index]
    probs = np.load(run / "bertopic_probability_matrix.npy", allow_pickle=False)
    initial = docs["bertopic_raw_topic"].ne(-1)
    topics = sorted(docs.loc[initial, "bertopic_raw_topic"].unique())
    if topics != list(range(len(topics))) or probs.shape != (len(docs), len(topics)):
        raise ValueError("Unexpected probability matrix or topic indices.")
    scores = probs.max(axis=1)
    best = probs.argmax(axis=1)
    np.testing.assert_allclose(scores, docs["bertopic_max_probability"], rtol=1e-8, atol=1e-12)
    themes = corpus["final_theme"]
    if themes.isna().any() or not set(themes) <= set(json.loads((ROOT / "ten_themes.json").read_text(encoding="utf-8"))):
        raise ValueError("Expected manually assigned final_theme values for all publications.")
    mapping = {}
    for t in topics:
        counts = themes.loc[docs["bertopic_raw_topic"].eq(t)].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            raise ValueError(f"Tied mapping for BERTopic topic {t}.")
        mapping[t] = counts.index[0]
    out = docs.loc[~initial, ["ut", "doi", "title"]].copy()
    out["highest_scoring_topic"] = best[~initial.to_numpy()]
    out["membership_score"] = scores[~initial.to_numpy()]
    out["meets_threshold"] = out["membership_score"].ge(0.20)
    expected = out["highest_scoring_topic"].where(out["meets_threshold"], -1)
    if not np.array_equal(expected, docs.loc[out.index, "selective"].astype(int)):
        raise ValueError("Highest scoring topics do not reproduce the selective run.")
    out["mapped_theme"] = out["highest_scoring_topic"].map(mapping)
    out["final_theme"] = themes
    out["agreement"] = out["mapped_theme"].eq(out["final_theme"])
    out.sort_index().to_csv(output / "no_threshold_agreement.csv", encoding="utf-8-sig")
    groups = {"at_or_above_0.20": out[out.meets_threshold],
              "below_0.20": out[~out.meets_threshold], "all_initial_outliers": out}
    pd.DataFrame([{"group": k, "records": len(v), "matches": int(v.agreement.sum()),
                   "agreement": v.agreement.mean()} for k, v in groups.items()]).to_csv(
                       output / "no_threshold_summary.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Ordered 735-record CSV with abstracts and final_theme.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compare-only", action="store_true", help="Reuse the retained and selective run outputs.")
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not a.compare_only:
        if a.input is None:
            p.error("--input is required unless --compare-only is used.")
        source = a.input.resolve()
        if len(pd.read_csv(source)) != 735:
            p.error("The published analysis requires 735 records.")
        def run(script, *args):
            subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)
        run("run_nmf_abstract.py", "--input", source, "--output-dir", out / "nmf",
            "--min-k", 10, "--max-k", 20)
        common = ["--input", source, "--device", a.device, "--random-state", 42]
        run("run_bertopic_abstract.py", *common, "--output-dir", out / "bertopic_retain", "--outlier-strategy", "none")
        run("run_bertopic_abstract.py", *common, "--output-dir", out / "bertopic_selective",
            "--outlier-strategy", "probabilities", "--outlier-threshold", 0.20)
    compare(out)
    check_reassignment(out)


if __name__ == "__main__":
    main()
