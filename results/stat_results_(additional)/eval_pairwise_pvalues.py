"""
Pairwise p-values for positional-bias results (no CLI parser).

This script expects you to edit the top-level PARAMETERS below to point to your
results CSV (the same format produced by eval_positional_bias.py). It will:

 - Read the CSV and count valid predicted answers (A/B/C/D).
 - For each pair of positions (A vs B, A vs C, ...), compute a chi-square
   goodness-of-fit test comparing observed [count1, count2] to equal expected
   counts (i.e., expected = total/2 for each).
 - Always output the raw p-value for every pair (not just significant ones).
 - Apply Bonferroni correction to determine significance and include that flag.
 - Print a readable table to stdout and save a CSV with all pairwise results.

Edit the PARAMETERS block below instead of using a CLI parser.

Requirements:
 - pandas
 - numpy
 - scipy

Run:
    python eval_pairwise_pvalues.py
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chisquare

# -----------------------------
# PARAMETERS (edit these values)
# -----------------------------
model = "llama3_8b-instruct-q6_K"
INPUT_CSV = f"2012-2020_ICT_DSE-{model}.csv"
ALPHA = 0.05
OUTPUT_CSV = f"pairwise_pvalues_output-{model}.csv"
VALID_LETTERS = ["A", "B", "C", "D"]
# -----------------------------

def load_results_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if "predicted_answer" not in df.columns:
        raise ValueError("Input CSV must contain a 'predicted_answer' column.")
    return df

def compute_choice_counts(df: pd.DataFrame) -> pd.Series:
    valid = df["predicted_answer"].astype(str).str.strip().str.upper()
    valid = valid[valid.isin(VALID_LETTERS)]
    counts = valid.value_counts().reindex(VALID_LETTERS, fill_value=0)
    return counts

def pairwise_pvalues(counts: pd.Series, alpha: float = 0.05):
    pairs = list(combinations(VALID_LETTERS, 2))
    results = []
    corrected_alpha = alpha / len(pairs) if len(pairs) > 0 else float("nan")

    for a, b in pairs:
        count_a = int(counts.get(a, 0))
        count_b = int(counts.get(b, 0))
        total = count_a + count_b

        if total == 0:
            chi2_stat = float("nan")
            p_value = float("nan")
        else:
            observed = np.array([count_a, count_b], dtype=float)
            expected = np.array([total / 2.0, total / 2.0], dtype=float)
            chi2_stat, p_value = chisquare(observed, f_exp=expected)

        significant = False
        if not np.isnan(p_value):
            significant = (p_value < corrected_alpha)

        results.append({
            "pos1": a,
            "pos2": b,
            "count_pos1": count_a,
            "count_pos2": count_b,
            "total_pair": total,
            "chi2_stat": chi2_stat,
            "p_value": p_value,
            "bonferroni_corrected_alpha": corrected_alpha,
            "significant_after_bonferroni": significant
        })

    return results

def pretty_print_results(counts: pd.Series, results: list):
    print("\nChoice counts (valid predicted answers only):")
    for letter in VALID_LETTERS:
        print(f"  {letter}: {counts[letter]}")

    print("\nPairwise comparisons (raw p-values shown for every pair):")
    header = f"{'Pair':7s} {'Count1':7s} {'Count2':7s} {'Total':7s} {'Chi2':8s} {'P-value':10s} {'Bonf.alpha':12s} {'Significant'}"
    print(header)
    print("-" * len(header))
    for r in results:
        pair_str = f"{r['pos1']}-{r['pos2']}"
        chi2_str = f"{r['chi2_stat']:.3f}" if not np.isnan(r["chi2_stat"]) else "nan"
        p_str = f"{r['p_value']:.6f}" if not np.isnan(r["p_value"]) else "nan"
        bonf_str = f"{r['bonferroni_corrected_alpha']:.6f}"
        sig_str = "YES" if r["significant_after_bonferroni"] else "NO"
        print(f"{pair_str:7s} {r['count_pos1']:7d} {r['count_pos2']:7d} {r['total_pair']:7d} {chi2_str:8s} {p_str:10s} {bonf_str:12s} {sig_str}")

def save_results_csv(results: list, output_path: str):
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\nPairwise results saved to: {output_path}")

def main():
    print("Pairwise p-value computation (no-arg mode).")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Family-wise alpha: {ALPHA} (Bonferroni will be applied)")
    print(f"Output CSV: {OUTPUT_CSV}")

    df = load_results_csv(INPUT_CSV)
    counts = compute_choice_counts(df)
    results = pairwise_pvalues(counts, alpha=ALPHA)

    pretty_print_results(counts, results)
    save_results_csv(results, OUTPUT_CSV)

if __name__ == "__main__":
    main()