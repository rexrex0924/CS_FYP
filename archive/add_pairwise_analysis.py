import pandas as pd
from scipy.stats import chisquare
from itertools import combinations
import os
from pathlib import Path

def calculate_pairwise_comparison(df: pd.DataFrame) -> list[str]:
    """Calculates the pairwise comparison stats and returns them as a list of strings."""
    
    report_lines = []
    # Filter for valid, parseable answers
    valid_responses = df[df["predicted_answer"].isin(["A", "B", "C", "D"])]
    
    if len(valid_responses) == 0:
        return ["PAIRWISE CHOICE COMPARISONS (Bonferroni-corrected):", "   No valid responses to analyze."]

    choice_counts = valid_responses["predicted_answer"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)

    report_lines.append("PAIRWISE CHOICE COMPARISONS (Bonferroni-corrected):")
    letters = ["A", "B", "C", "D"]
    comparisons = list(combinations(letters, 2))
    alpha = 0.05
    # Bonferroni correction for multiple comparisons
    corrected_alpha = alpha / len(comparisons)
    significant_pairs = 0
    
    for pos1, pos2 in comparisons:
        count1 = choice_counts[pos1]
        count2 = choice_counts[pos2]
        
        # A chi-square test on two counts tests if they are significantly different from the expected frequency (their mean).
        if count1 + count2 > 0:
            _, p_pair = chisquare([count1, count2])
            if p_pair < corrected_alpha:
                significant_pairs += 1
                report_lines.append(f"   - {pos1} vs {pos2}: Significant difference (p={p_pair:.4f} < {corrected_alpha:.4f})")

    if significant_pairs == 0:
        report_lines.append("   No pairs showed significant differences in choice frequency.")
        
    return report_lines


def process_stat_files(stat_dir: str, csv_dir: str):
    """
    Iterates through analysis text files, calculates pairwise comparisons from corresponding CSVs,
    and inserts the results into the text files.
    """
    stat_path = Path(stat_dir)
    csv_path = Path(csv_dir)

    if not stat_path.is_dir() or not csv_path.is_dir():
        print(f"Error: One of the directories does not exist: '{stat_dir}', '{csv_dir}'")
        return

    for stat_file in stat_path.glob("*_analysis.txt"):
        print(f"Processing {stat_file.name}...")

        # Construct the corresponding CSV file name
        csv_filename = stat_file.name.replace("_analysis.txt", ".csv")
        csv_file = csv_path / csv_filename
        
        if not csv_file.exists():
            print(f"  - SKIPPING: Corresponding CSV file not found: {csv_file.name}")
            continue

        try:
            # Read original report
            with open(stat_file, 'r', encoding='utf-8') as f:
                original_lines = f.read().splitlines()

            # Check if pairwise comparison section already exists
            if any("PAIRWISE CHOICE COMPARISONS" in line for line in original_lines):
                print("  - SKIPPING: Pairwise comparison section already exists.")
                continue

            # Load data and calculate pairwise comparison
            df = pd.read_csv(csv_file)
            pairwise_lines = calculate_pairwise_comparison(df)
            
            # Find the insertion point: right before the "ACCURACY BY CORRECT ANSWER POSITION" section
            insertion_point = -1
            for i, line in enumerate(original_lines):
                if "ACCURACY BY CORRECT ANSWER POSITION" in line:
                    insertion_point = i
                    break
            
            if insertion_point != -1:
                # Insert the new section with a blank line for spacing
                final_lines = (
                    original_lines[:insertion_point] + 
                    [""] + 
                    pairwise_lines + 
                    original_lines[insertion_point:]
                )
                print("  - SUCCESS: Pairwise comparison added.")
            else:
                # If the anchor text isn't found, append to the end
                final_lines = original_lines + [""] + pairwise_lines
                print("  - WARNING: 'ACCURACY' section not found. Appending to end of file.")

            # Write the updated content back to the file
            with open(stat_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(final_lines) + "\n")

        except Exception as e:
            print(f"  - ERROR: Failed to process file {stat_file.name}. Reason: {e}")


if __name__ == "__main__":
    # Define the directories where your results are stored
    stat_results_dir = "results/stat_results"
    csv_results_dir = "results/csv_results"
    
    print(f"Starting batch update for files in '{stat_results_dir}'...")
    process_stat_files(stat_results_dir, csv_results_dir)
    print("\nBatch update complete.")
