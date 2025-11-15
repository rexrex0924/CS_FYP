"""
Re-analyze Positional Bias Statistics from Existing CSV Results

This script reads previously generated CSV files and performs comprehensive 
statistical analysis, showing ALL raw values and p-values regardless of 
whether hypotheses are rejected or not.

Usage:
    python reanalyze_bias_stats.py --input results/csv_results/college_cs-gemma3_4b_sampling_n15.csv
    python reanalyze_bias_stats.py --input results/csv_results/ --batch
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency


def analyze_results_comprehensive(df: pd.DataFrame, model: str, dataset_name: str, 
                                  output_dir: Path) -> None:
    """
    Comprehensive statistical analysis with ALL raw values and p-values shown.
    
    This function ALWAYS shows:
    - Raw counts and percentages
    - Chi-square statistics AND p-values
    - All pairwise comparisons (not just significant ones)
    - Complete contingency tables
    - Position bias scores
    """
    
    output_filename = f"{dataset_name}_detailed_analysis.txt"
    analysis_file_path = output_dir / output_filename
    report_lines = []

    def _log(message):
        print(message)
        report_lines.append(str(message))

    _log("=" * 80)
    _log("COMPREHENSIVE POSITIONAL BIAS ANALYSIS")
    _log("=" * 80)
    _log(f"Model: {model}")
    _log(f"Dataset: {dataset_name}")
    _log(f"Total Evaluations: {len(df)}")
    _log("=" * 80)
    
    # Filter out failed responses
    valid_responses = df[df["predicted_answer"].isin(["A", "B", "C", "D"])].copy()
    failed_responses = len(df) - len(valid_responses)
    
    _log(f"\nDATA QUALITY:")
    _log(f"  Valid responses: {len(valid_responses)} ({len(valid_responses)/len(df)*100:.1f}%)")
    _log(f"  Failed responses: {failed_responses} ({failed_responses/len(df)*100:.1f}%)")
    
    if len(valid_responses) == 0:
        _log("\nERROR: No valid responses to analyze")
        with open(analysis_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\nAnalysis report saved to: {analysis_file_path}")
        return
    
    # ========================================================================
    # 1. CHOICE DISTRIBUTION ANALYSIS
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("1. CHOICE DISTRIBUTION (Predicted Answers)")
    _log("=" * 80)
    
    choice_counts = valid_responses["predicted_answer"].value_counts().reindex(
        ["A", "B", "C", "D"], fill_value=0
    )
    total_valid = len(valid_responses)
    
    _log(f"\nRaw Counts and Percentages (n={total_valid}):")
    _log(f"{'Choice':<10} {'Count':<10} {'Percentage':<15} {'Expected (25%)':<20}")
    _log("-" * 60)
    
    for letter in ["A", "B", "C", "D"]:
        count = choice_counts[letter]
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        expected = total_valid / 4
        deviation = count - expected
        _log(f"{letter:<10} {count:<10} {percentage:>6.2f}% {' ' * 8} {expected:>6.1f} (Δ={deviation:+.1f})")
    
    # ========================================================================
    # 2. CHI-SQUARE TEST: Choice Distribution vs Uniform
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("2. CHI-SQUARE TEST: Choice Distribution vs Uniform (25/25/25/25)")
    _log("=" * 80)
    
    expected_per_choice = total_valid / 4
    expected = [expected_per_choice] * 4
    chi2_stat, p_value = chisquare(choice_counts.values, f_exp=expected)
    
    _log(f"\nObserved Frequencies: A={choice_counts['A']}, B={choice_counts['B']}, "
         f"C={choice_counts['C']}, D={choice_counts['D']}")
    _log(f"Expected Frequencies: A={expected_per_choice:.1f}, B={expected_per_choice:.1f}, "
         f"C={expected_per_choice:.1f}, D={expected_per_choice:.1f}")
    _log(f"\nChi-square Statistic (χ²): {chi2_stat:.4f}")
    _log(f"Degrees of Freedom: 3")
    _log(f"P-value: {p_value:.6f}")
    _log(f"\nInterpretation:")
    if p_value < 0.001:
        _log(f"  *** Highly significant deviation from uniform (p < 0.001)")
    elif p_value < 0.01:
        _log(f"  ** Significant deviation from uniform (p < 0.01)")
    elif p_value < 0.05:
        _log(f"  * Significant deviation from uniform (p < 0.05)")
    else:
        _log(f"  No significant deviation from uniform (p >= 0.05)")
    _log(f"  Effect Size (Cramér's V): {np.sqrt(chi2_stat / (total_valid * 3)):.4f}")
    
    # ========================================================================
    # 3. PAIRWISE CHOICE COMPARISONS (All Pairs)
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("3. PAIRWISE CHOICE COMPARISONS (Bonferroni-corrected)")
    _log("=" * 80)
    
    letters = ["A", "B", "C", "D"]
    comparisons = list(combinations(letters, 2))
    alpha = 0.05
    corrected_alpha = alpha / len(comparisons)
    
    _log(f"\nTotal Comparisons: {len(comparisons)}")
    _log(f"Significance Level (α): {alpha}")
    _log(f"Bonferroni-corrected α: {corrected_alpha:.4f}")
    _log(f"\n{'Comparison':<15} {'Count 1':<10} {'Count 2':<10} {'χ² Statistic':<15} "
         f"{'P-value':<12} {'Significant?':<15}")
    _log("-" * 85)
    
    significant_pairs = 0
    for pos1, pos2 in comparisons:
        count1 = choice_counts[pos1]
        count2 = choice_counts[pos2]
        
        # Chi-square test for two proportions
        if count1 + count2 > 0:
            chi2_pair, p_pair = chisquare([count1, count2])
            is_significant = "YES ***" if p_pair < corrected_alpha else "No"
            if p_pair < corrected_alpha:
                significant_pairs += 1
            
            _log(f"{pos1} vs {pos2:<10} {count1:<10} {count2:<10} {chi2_pair:<15.4f} "
                 f"{p_pair:<12.6f} {is_significant:<15}")
        else:
            _log(f"{pos1} vs {pos2:<10} {count1:<10} {count2:<10} {'N/A':<15} "
                 f"{'N/A':<12} {'N/A':<15}")
    
    _log(f"\nSummary: {significant_pairs}/{len(comparisons)} pairs show significant differences")
    
    # ========================================================================
    # 4. ACCURACY BY CORRECT ANSWER POSITION
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("4. ACCURACY BY CORRECT ANSWER POSITION")
    _log("=" * 80)
    
    accuracy_by_position = valid_responses.groupby("correct_position")["is_correct"].agg([
        ('count', 'count'),
        ('correct', 'sum'),
        ('accuracy', 'mean')
    ])
    
    overall_accuracy = valid_responses["is_correct"].mean()
    
    _log(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    _log(f"\n{'Position':<10} {'Total':<10} {'Correct':<10} {'Accuracy':<12} "
         f"{'Deviation':<15}")
    _log("-" * 65)
    
    for letter in ["A", "B", "C", "D"]:
        if letter in accuracy_by_position.index:
            count = int(accuracy_by_position.loc[letter, "count"])
            correct = int(accuracy_by_position.loc[letter, "correct"])
            acc = accuracy_by_position.loc[letter, "accuracy"]
            diff = acc - overall_accuracy
            _log(f"{letter:<10} {count:<10} {correct:<10} {acc:<12.4f} {diff:>+7.4f}")
        else:
            _log(f"{letter:<10} {'0':<10} {'0':<10} {'N/A':<12} {'N/A':<15}")
    
    # ========================================================================
    # 5. CHI-SQUARE TEST: Accuracy vs Position Independence
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("5. CHI-SQUARE TEST: Accuracy vs Position (Independence)")
    _log("=" * 80)
    
    try:
        contingency_table = pd.crosstab(
            valid_responses['correct_position'], 
            valid_responses['is_correct']
        )
        contingency_table = contingency_table.reindex(["A", "B", "C", "D"], fill_value=0)
        
        # Ensure both columns exist
        if 0 not in contingency_table.columns:
            contingency_table[0] = 0
        if 1 not in contingency_table.columns:
            contingency_table[1] = 0
        
        contingency_table = contingency_table.rename(
            columns={0: 'Incorrect', 1: 'Correct'}
        )[['Correct', 'Incorrect']]
        
        _log("\nContingency Table:")
        _log(contingency_table.to_string())
        
        # Perform chi-square test
        if (contingency_table.sum().sum() > 0 and 
            not (contingency_table.sum(axis=0) == 0).any() and 
            not (contingency_table.sum(axis=1) == 0).any()):
            
            chi2_acc, p_acc, dof, expected_freq = chi2_contingency(contingency_table)
            
            _log(f"\nExpected Frequencies (under independence):")
            expected_df = pd.DataFrame(
                expected_freq, 
                index=["A", "B", "C", "D"],
                columns=["Correct", "Incorrect"]
            )
            _log(expected_df.to_string())
            
            _log(f"\nChi-square Statistic (χ²): {chi2_acc:.4f}")
            _log(f"Degrees of Freedom: {dof}")
            _log(f"P-value: {p_acc:.6f}")
            _log(f"\nInterpretation:")
            if p_acc < 0.001:
                _log(f"  *** Highly significant relationship (p < 0.001)")
                _log(f"      Accuracy STRONGLY depends on answer position")
            elif p_acc < 0.01:
                _log(f"  ** Significant relationship (p < 0.01)")
                _log(f"     Accuracy depends on answer position")
            elif p_acc < 0.05:
                _log(f"  * Significant relationship (p < 0.05)")
                _log(f"    Accuracy may depend on answer position")
            else:
                _log(f"  No significant relationship (p >= 0.05)")
                _log(f"  Accuracy appears independent of answer position")
            
            _log(f"\nEffect Size (Cramér's V): {np.sqrt(chi2_acc / (total_valid * min(dof, 1))):.4f}")
        else:
            _log("\nInsufficient data diversity for chi-square test")
            _log("(Some categories have zero observations)")
            
    except Exception as e:
        _log(f"\nError performing chi-square test: {e}")
    
    # ========================================================================
    # 6. POSITION BIAS SCORE
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("6. POSITION BIAS SCORE")
    _log("=" * 80)
    
    choice_percentages = choice_counts.values / total_valid * 100
    position_bias_score = np.std(choice_percentages)
    
    _log(f"\nChoice Percentages: A={choice_percentages[0]:.2f}%, "
         f"B={choice_percentages[1]:.2f}%, "
         f"C={choice_percentages[2]:.2f}%, "
         f"D={choice_percentages[3]:.2f}%")
    _log(f"Expected (Uniform): 25.00%, 25.00%, 25.00%, 25.00%")
    _log(f"\nPosition Bias Score (Std Dev): {position_bias_score:.4f}")
    _log(f"\nInterpretation:")
    if position_bias_score < 2:
        _log(f"  Very low bias (score < 2)")
    elif position_bias_score < 5:
        _log(f"  Low to moderate bias (2 <= score < 5)")
    elif position_bias_score < 10:
        _log(f"  Moderate to high bias (5 <= score < 10)")
    else:
        _log(f"  High bias (score >= 10)")
    
    # ========================================================================
    # 7. RECALL BY POSITION (PriDe-style metric)
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("7. RECALL BY POSITION (Recall Standard Deviation)")
    _log("=" * 80)
    
    positions = ['A', 'B', 'C', 'D']
    recalls = []
    recall_details = []
    
    _log(f"\n{'Position':<10} {'Questions':<12} {'Recalled':<12} {'Recall Rate':<15}")
    _log("-" * 55)
    
    for pos in positions:
        pos_mask = valid_responses['correct_position'] == pos
        n_questions = pos_mask.sum()
        if n_questions > 0:
            n_correct = (valid_responses[pos_mask]['predicted_answer'] == pos).sum()
            recall = n_correct / n_questions
            recalls.append(recall)
            recall_details.append((pos, n_questions, n_correct, recall))
            _log(f"{pos:<10} {n_questions:<12} {n_correct:<12} {recall:<15.4f} ({recall*100:.2f}%)")
        else:
            recalls.append(0.0)
            recall_details.append((pos, 0, 0, 0.0))
            _log(f"{pos:<10} {0:<12} {0:<12} {'N/A':<15}")
    
    recall_std = np.std(recalls) * 100  # Report as percentage
    recall_mean = np.mean(recalls) * 100
    
    _log(f"\nRecall Statistics:")
    _log(f"  Mean Recall: {recall_mean:.2f}%")
    _log(f"  Recall Std Dev (RStd): {recall_std:.2f}%")
    _log(f"\nInterpretation:")
    _log(f"  Lower RStd indicates more balanced recall across positions")
    _log(f"  Higher RStd indicates position-dependent accuracy")
    
    # ========================================================================
    # 8. SAMPLE SIZE AND PROBABILITY ESTIMATION QUALITY
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("8. SAMPLE SIZE AND ESTIMATION QUALITY")
    _log("=" * 80)
    
    if 'total_valid_samples' in valid_responses.columns:
        sample_sizes = valid_responses['total_valid_samples']
        _log(f"\nSamples per evaluation:")
        _log(f"  Mean: {sample_sizes.mean():.2f}")
        _log(f"  Median: {sample_sizes.median():.0f}")
        _log(f"  Min: {sample_sizes.min():.0f}")
        _log(f"  Max: {sample_sizes.max():.0f}")
        _log(f"  Std Dev: {sample_sizes.std():.2f}")
        
        # Margin of error estimation
        avg_n = sample_sizes.mean()
        margin_of_error = 1.96 * np.sqrt(0.25 * 0.75 / avg_n) * 100  # 95% CI for p=0.25
        _log(f"\nEstimated Margin of Error (95% CI): ±{margin_of_error:.2f}%")
        _log(f"  (Based on average n={avg_n:.1f} samples, assuming p=0.25)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    _log("\n" + "=" * 80)
    _log("SUMMARY")
    _log("=" * 80)
    _log(f"\n✓ Overall Accuracy: {overall_accuracy:.4f}")
    _log(f"✓ Position Bias Score: {position_bias_score:.4f}")
    _log(f"✓ Recall Std Dev: {recall_std:.2f}%")
    _log(f"✓ Chi² (Distribution): {chi2_stat:.4f} (p={p_value:.6f})")
    if 'chi2_acc' in locals():
        _log(f"✓ Chi² (Accuracy vs Position): {chi2_acc:.4f} (p={p_acc:.6f})")
    
    _log("\n" + "=" * 80)
    
    # Save report
    with open(analysis_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ Detailed analysis report saved to: {analysis_file_path}")


def process_single_file(csv_path: Path, output_dir: Path) -> None:
    """Process a single CSV file."""
    print(f"\n{'='*80}")
    print(f"Processing: {csv_path.name}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Extract model name from CSV or filename
        if 'model' in df.columns and len(df) > 0:
            model = df['model'].iloc[0]
        else:
            model = "Unknown"
        
        # Extract dataset name from filename
        dataset_name = csv_path.stem
        
        analyze_results_comprehensive(df, model, dataset_name, output_dir)
        
    except Exception as e:
        print(f"❌ Error processing {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()


def process_batch(input_dir: Path, output_dir: Path) -> None:
    """Process all CSV files in a directory."""
    csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
        process_single_file(csv_file, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Batch processing complete! Processed {len(csv_files)} files.")
    print(f"✅ Reports saved to: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-analyze positional bias statistics from existing CSV results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single CSV file
  python reanalyze_bias_stats.py --input results/csv_results/college_cs-gemma3_4b_sampling_n15.csv

  # Batch process all CSV files in a directory
  python reanalyze_bias_stats.py --input results/csv_results/ --batch

  # Specify custom output directory
  python reanalyze_bias_stats.py --input results/csv_results/ --batch --output results/detailed_stats/
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file or directory containing CSV files"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all CSV files in the input directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/detailed_stat_results",
        help="Output directory for analysis reports (default: results/detailed_stat_results)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("POSITIONAL BIAS STATISTICS RE-ANALYZER")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Batch' if args.batch else 'Single File'}")
    print("="*80)
    
    if args.batch:
        if not input_path.is_dir():
            print(f"❌ Error: {input_path} is not a directory")
            print("   Use --batch only with a directory path")
            return
        process_batch(input_path, output_dir)
    else:
        if not input_path.is_file():
            print(f"❌ Error: {input_path} is not a file")
            print("   For batch processing, use --batch flag")
            return
        process_single_file(input_path, output_dir)


if __name__ == "__main__":
    main()

