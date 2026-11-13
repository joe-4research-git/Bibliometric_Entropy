# Reproducibility code for the 735-publication review

Install dependencies with `python -m pip install -r requirements.txt`.

1. Initial screening: `python 01_wos_query_and_rule_screening.py --input wos_entropy_core.csv --output-dir screening --strict`. Supply the original WoS full-record export in a tabular format. This script ends at initial automated screening; see Supplementary Materials 2–4 for audit, targeted re-screening and adjudication.
2. Topic models: `python 02_nmf_and_bertopic.py --input corpus_735.csv --output-dir models`. Supply the ordered 735-publication abstract input used for modeling. S2 does not contain abstracts and cannot substitute for this input. The helper scripts and ten-theme vocabulary must stay with the entry point. BERTopic may need to download the pretrained embedding model. Use `--compare-only` with the original retained and selective output directories to reproduce the comparisons without refitting. Final theme labels are used only for the post-model comparison; they are not modeling features. Expert adjudication is not automated.
3. Metrics: `python 03_compute_metrics_from_s2.py --input "Supplementary Material 1 - Language Signals, Corpus Records, and Source Mapping.xlsx" --output-dir metrics --strict`. Use the final S2 sheet with harmonized source families. Adjust the filename if needed. HHI uses the five complete five-year periods; theme-by-measure summaries use the full corpus.

The ZIP contains code and dependency specifications. It does not include WoS exports, the abstract input, or fitted models.
