
#!/usr/bin/env python3
"""run_anova_tukey.py
---------------------------------
Compute a two‑way ANOVA and Tukey HSD on OpenTas parsed scores.

• Works with any row count per JSONL file (e.g., 7 920 or 1 800).
• Requires eight JSONL files covering four model types (3B‑base, 3B‑FT,
  7B‑base, 7B‑FT) × four context framings (NON, WRONG, CORRECT, PROMPT).
• Each JSON line must contain the key ``score_vs_pred_strict``.
• Edit only the FILES list and OUTPUT_PATH below.

Dependencies
------------
    pip install pandas statsmodels scipy
"""

import json
import sys
import pathlib
from typing import List, Dict

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm  
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --------------------------------------------------------------------------
# >>>>>> USER CONFIGURATION <<<<<<
#
# Provide exactly 16 dicts – one per experiment – or keep the 8 “big”
# experiments if each JSONL already aggregates across prompts.  The script
# will gracefully warn if any Context × Model combination is missing.
#
# Keys:
#   path    : str – absolute or relative path to the *.jsonl* file
#   model   : str – one of {"3B-base", "3B-FT", "7B-base", "7B-FT"}
#   context : str – one of {"NON", "WRONG", "CORRECT", "PROMPT"}
#
FILES: List[Dict[str, str]] = [
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/non_labeled/standart_model/normal_data/Normal_test_3B_openTAS_NON_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",       "model": "3B-base", "context": "NON"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Wrongly_labeled/standart_model/normal_data/Normal_test_3B_openTAS_wrongly_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",     "model": "3B-base", "context": "WRONG"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Correctly_labeled/standart_model/normal_data/Normal_test_3B_openTAS_correctly_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",   "model": "3B-base", "context": "CORRECT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Prompt_and_correct/standart_model/normal_data/Normal_test_3B_openTAS_PROMPT_and_Correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",    "model": "3B-base", "context": "PROMPT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/non_labeled/fine_tuned_model/Normal_data/Normal_test_3B_openTAS_NONlabeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",         "model": "3B-FT",   "context": "NON"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Wrongly_labeled/fine_tuned_model/Normal_data/Normal_test_3B_openTAS_wrongly_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",       "model": "3B-FT",   "context": "WRONG"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Correctly_labeled/Fine_tuned_model/Normal_data/Normal_test_3B_openTAS_Correctlabeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",     "model": "3B-FT",   "context": "CORRECT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/3BopenTAS/Prompt_and_correct/fine_tuned_model/Normal_data/Normal_test_3B_openTAS_PROMPT_and_Correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",      "model": "3B-FT",   "context": "PROMPT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/standart_model/normal_data/Normal_test_7B_openTAS_NON_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",       "model": "7B-base", "context": "NON"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Wrongly_labeled/standart_model/normal_data/Normal_test_7B_openTAS_Wrongly_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",     "model": "7B-base", "context": "WRONG"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Correctly_labeled/standart_model/normal_data/Normal_test_7B_openTAS_correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",   "model": "7B-base", "context": "CORRECT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Prompt_and_correct/standart_model/normal_data/Normal_test_7B_openTAS_PROMPT_and_correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",    "model": "7B-base", "context": "PROMPT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/Normal_data/Normal_test_7B_openTAS_NON_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",         "model": "7B-FT",   "context": "NON"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Wrongly_labeled/fine_tuned_model/Normal_data/Normal_test_7B_openTAS_Wrongly_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",       "model": "7B-FT",   "context": "WRONG"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Correctly_labeled/fine_tuned_model/Normal_data/Normal_test_7B_openTAS_Correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",     "model": "7B-FT",   "context": "CORRECT"},
    {"path": "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/Prompt_and_correct/fine_tuned_model/Normal_data/Normal_test_7B_openTAS_PROMPT_and_Correct_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl",      "model": "7B-FT",   "context": "PROMPT"},
]

# Plain‑text output for the report
OUTPUT_PATH = pathlib.Path("/Users/yer/scriptie/anova_tukey_report.txt")
# --------------------------------------------------------------------------



def load_scores(spec: Dict[str, str]):
    """Return list of dicts with Model, Context, Score for one JSONL file."""
    fpath = pathlib.Path(spec["path"])
    if not fpath.is_file():
        sys.exit(f"File not found: {fpath}")
    rows = []
    with fpath.open() as fh:
        for line in fh:
            try:
                score = json.loads(line)["score_vs_pred_strict"]
            except (KeyError, json.JSONDecodeError):
                continue
            rows.append({"Model": spec["model"],
                         "Context": spec["context"].upper(),
                         "Score": score})
    if not rows:
        print(f"⚠️  No valid rows read from {fpath}")
    return rows


def main():
    # ------------------------------------------------------------------ ingest
    records = []
    for spec in FILES:
        records.extend(load_scores(spec))
    if not records:
        sys.exit("No data loaded. Check FILES list and JSONL content.")

    df = pd.DataFrame(records)
    print("\nRow count per cell:\n",
          df.groupby(["Model", "Context"]).size().unstack(fill_value=0), "\n")

    # ------------------------------------------------------------- two-way ANOVA
    ols_fit = smf.ols("Score ~ C(Context) * C(Model)", data=df).fit()
    anova_tbl = sm.stats.anova_lm(ols_fit, typ=2)

    # --------------------------------------------------------- Tukey contrasts
    tukey_frames = []
    for mdl in sorted(df["Model"].unique()):
        sub = df[df["Model"] == mdl]
        res = pairwise_tukeyhsd(sub["Score"], sub["Context"], alpha=0.05)
        hdr, *rows = res.summary().data      # compatible across statsmodels versions
        tdf = pd.DataFrame(rows, columns=hdr)
        tdf.insert(0, "Model", mdl)
        tukey_frames.append(tdf)
    tukey_df = pd.concat(tukey_frames, ignore_index=True)

    # ------------------------------------------------------------------ report
    with OUTPUT_PATH.open("w") as out:
        out.write("==== TWO-WAY ANOVA (Score ~ Context * Model) ====\n")
        out.write(anova_tbl.to_string())
        out.write("\n\n==== Tukey HSD (Context contrasts within each Model) ====\n")
        out.write(tukey_df.to_string(index=False))

    print(f"✔ Report written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()