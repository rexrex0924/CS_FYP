"""
Batch Statistical Analysis for CSV Results
Processes all CSV files from a source directory and generates detailed 
statistical analysis text files, organized by dataset into a destination directory.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chisquare, chi2_contingency
from itertools import combinations
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')


def analyze_csv(csv_path: Path) -> dict:
    """
    Analyzes a single CSV file and computes all relevant statistical metrics.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   - Error reading {csv_path.name}: {e}")
        return None

    # --- Basic Statistics ---
    total_questions = len(df)
    if total_questions == 0:
        print(f"   - Skipping {csv_path.name}: No data found.")
        return None
        
    choice_counts = Counter(df['predicted_answer'].astype(str).str.upper().str.strip())
    
    # Ensure all choices A, B, C, D are present
    for letter in ['A', 'B', 'C', 'D']:
        choice_counts[letter] = choice_counts.get(letter, 0)

    accuracy = df['is_correct'].sum() / total_questions * 100

    # --- Chi-Square Test for Uniform Distribution (Goodness-of-Fit) ---
    observed = [choice_counts['A'], choice_counts['B'], choice_counts['C'], choice_counts['D']]
    expected_uniform = [total_questions / 4] * 4
    chi2_uniform, p_uniform = chisquare(observed, f_exp=expected_uniform)

    # --- Position Bias Score ---
    percentages = [(count / total_questions * 100) for count in observed]
    bias_score = np.std(percentages)

    # --- Chi-Square Test for Independence (Accuracy vs Position) ---
    # Convert is_correct to string labels ('Correct'/'Incorrect')
    # This handles both boolean and integer values properly
    df['outcome'] = df['is_correct'].astype(bool).map({True: 'Correct', False: 'Incorrect'})
    contingency_table = pd.crosstab(df['correct_position'], df['outcome'])
    
    # Ensure table has both 'Correct' and 'Incorrect' columns for consistency
    if 'Correct' not in contingency_table.columns: contingency_table['Correct'] = 0
    if 'Incorrect' not in contingency_table.columns: contingency_table['Incorrect'] = 0
    contingency_table = contingency_table[['Correct', 'Incorrect']] # Ensure order
    
    # Ensure all positions A, B, C, D are present as rows
    for pos in ['A', 'B', 'C', 'D']:
        if pos not in contingency_table.index:
            contingency_table.loc[pos] = [0, 0]
    contingency_table = contingency_table.loc[['A', 'B', 'C', 'D']]
    
    chi2_indep, p_indep, dof_indep, _ = chi2_contingency(contingency_table)

    # --- Recall Standard Deviation (RStd) ---
    recalls = []
    for pos in ['A', 'B', 'C', 'D']:
        pos_mask = df['correct_position'] == pos
        if pos_mask.sum() > 0:
            recall = (df.loc[pos_mask, 'predicted_answer'] == pos).mean()
            recalls.append(recall)
        else:
            recalls.append(0.0)
    recall_std = np.std(recalls) * 100

    # --- Pairwise Comparisons ---
    pairwise_results = []
    letters = ['A', 'B', 'C', 'D']
    comparisons = list(combinations(letters, 2))
    corrected_alpha = 0.05 / len(comparisons)
    
    for pos1, pos2 in comparisons:
        count1 = choice_counts[pos1]
        count2 = choice_counts[pos2]
        
        if count1 + count2 > 0:
            stat_pair, p_pair = chisquare([count1, count2])
            is_significant = p_pair < corrected_alpha
            pairwise_results.append({
                'pair': f"{pos1} vs {pos2}", 'chi2': stat_pair, 'p_value': p_pair, 'significant': is_significant
            })
        else:
            pairwise_results.append({
                'pair': f"{pos1} vs {pos2}", 'chi2': 0.0, 'p_value': 1.0, 'significant': False
            })

    # --- Accuracy by Position ---
    accuracy_by_position = df.groupby('correct_position')['is_correct'].mean().apply(lambda x: x * 100).to_dict()
    for pos in ['A', 'B', 'C', 'D']:
        accuracy_by_position.setdefault(pos, 0.0)

    # --- Consistency Score ---
    # Calculate the percentage of questions where the model chooses the same *content* across all permutations.
    # We map the predicted letter back to the original option index using the permutation logic.
    def get_original_choice(row):
        try:
            perm_idx = int(row['permutation_idx'])
            pred = str(row['predicted_answer']).strip().upper()
            if pred not in ['A', 'B', 'C', 'D']:
                return None
            
            letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            reverse_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            
            pred_idx = letter_map[pred]
            shift = perm_idx % 4
            
            # cyclic_items[i] corresponds to original_items[(shift + i) % 4]
            # So the predicted option at position i is the original option at (shift + i) % 4
            orig_idx = (shift + pred_idx) % 4
            return reverse_map[orig_idx]
        except (ValueError, KeyError, TypeError):
            return None

    # Use question_id if available, otherwise try id
    id_col = 'question_id' if 'question_id' in df.columns else 'id'
    
    if id_col in df.columns and 'permutation_idx' in df.columns:
        df['original_choice'] = df.apply(get_original_choice, axis=1)
        
        # Group by question ID and count unique original choices
        # We filter out rows where original_choice is None
        valid_consistency_df = df.dropna(subset=['original_choice'])
        
        if not valid_consistency_df.empty:
            consistency_counts = valid_consistency_df.groupby(id_col)['original_choice'].nunique()
            consistent_questions = (consistency_counts == 1).sum()
            total_unique_questions = len(consistency_counts)
            consistency_score = (consistent_questions / total_unique_questions * 100) if total_unique_questions > 0 else 0.0
        else:
            consistency_score = 0.0
    else:
        consistency_score = 0.0

    return {
        'total_questions': total_questions, 'overall_accuracy': accuracy, 'choice_counts': choice_counts,
        'chi2_uniform': chi2_uniform, 'p_uniform': p_uniform, 'bias_score': bias_score,
        'chi2_indep': chi2_indep, 'p_indep': p_indep, 'recall_std': recall_std, 'recalls': recalls,
        'pairwise': pairwise_results, 'corrected_alpha': corrected_alpha, 'accuracy_by_position': accuracy_by_position,
        'consistency_score': consistency_score
    }


def format_analysis_report(results: dict, model_name: str, dataset_name: str) -> str:
    """
    Formats the analysis results into a text report matching the original format.
    """
    lines = [
        "=" * 80,
        f"STATISTICAL ANALYSIS: {dataset_name} - {model_name}",
        "=" * 80,
        "",
        "BASIC STATISTICS:",
        f"   - Total Questions: {results['total_questions']}",
        f"   - Overall Accuracy: {results['overall_accuracy']:.2f}%",
        "",
        "CHOICE DISTRIBUTION:",
    ]
    for letter in ['A', 'B', 'C', 'D']:
        count = results['choice_counts'][letter]
        pct = (count / results['total_questions'] * 100) if results['total_questions'] > 0 else 0
        lines.append(f"   - {letter}: {count} ({pct:.2f}%)")
    
    lines.extend([
        "",
        "CHI-SQUARE TEST vs Uniform Distribution:",
        f"   - Chi-square statistic: {results['chi2_uniform']:.2f}",
        f"   - p-value: {results['p_uniform']:.6f}",
        "   - Degrees of freedom: 3",
        f"   - Result: {'SIGNIFICANT bias detected (p < 0.05)' if results['p_uniform'] < 0.05 else 'No significant bias (p >= 0.05)'}",
        "",
        "POSITION BIAS SCORE:",
        f"   - Standard deviation of choice percentages: {results['bias_score']:.2f}",
        "",
        "CHI-SQUARE TEST for Accuracy vs Position:",
        f"   - Chi-square statistic: {results['chi2_indep']:.2f}",
        f"   - p-value: {results['p_indep']:.6f}",
        "   - Degrees of freedom: 3",
        f"   - Result: {'Accuracy IS dependent on position (p < 0.05)' if results['p_indep'] < 0.05 else 'Accuracy is independent of position (p >= 0.05)'}",
        "",
        "CONSISTENCY SCORE:",
        f"   - Consistency Rate: {results.get('consistency_score', 0.0):.2f}% (Questions with same content answer across permutations)",
        "",
        "RECALL STANDARD DEVIATION (RStd):",
        "   - Recall per position:",
    ])
    for i, letter in enumerate(['A', 'B', 'C', 'D']):
        lines.append(f"     {letter}: {results['recalls'][i]:.4f}")
    lines.append(f"   - Standard Deviation: {results['recall_std']:.2f}%")
    
    lines.extend([
        "",
        "PAIRWISE CHOICE COMPARISONS (Bonferroni-corrected):",
        f"   - Corrected significance level (alpha/k): {results['corrected_alpha']:.4f}",
    ])
    for pair_data in results['pairwise']:
        sig_marker = " (Significant)" if pair_data['significant'] else ""
        lines.append(f"   - {pair_data['pair']}: chi2={pair_data['chi2']:.2f}, p={pair_data['p_value']:.4f}{sig_marker}")
        
    lines.extend([
        "",
        "ACCURACY BY CORRECT ANSWER POSITION:",
    ])
    for letter in ['A', 'B', 'C', 'D']:
        lines.append(f"   - Position {letter}: {results['accuracy_by_position'][letter]:.2f}%")
    
    lines.extend(["", "=" * 80])
    return "\n".join(lines)


def create_pairwise_heatmap(dataset_name: str, models_data: dict, output_dir: Path):
    """
    Creates a heatmap visualization of pairwise p-values for all models in a dataset.
    
    Args:
        dataset_name: Name of the dataset
        models_data: Dict mapping model_name -> analysis_results
        output_dir: Directory to save the heatmap
    """
    # Use all models provided, without filtering by bias score.
    models_to_plot = models_data
    
    if not models_to_plot:
        print(f"   - Skipping heatmap for {dataset_name}: No models to plot.")
        return
    
    n_models = len(models_to_plot)
    ncols = min(3, n_models)  # Max 3 columns
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    corrected_alpha = 0.0083
    
    for idx, (model_name, results) in enumerate(sorted(models_to_plot.items())):
        ax = axes[idx]
        
        # Create 4x4 matrix for the heatmap (upper triangle only)
        matrix = np.ones((4, 4)) * np.nan  # Use NaN for diagonal and lower triangle
        labels = np.full((4, 4), '', dtype=object)
        significant_mask = np.zeros((4, 4), dtype=bool)  # Track which cells are significant
        
        # Fill upper triangle with p-values
        for pair_data in results['pairwise']:
            pair = pair_data['pair']
            pos1, pos2 = pair.split(' vs ')
            i, j = ['A', 'B', 'C', 'D'].index(pos1), ['A', 'B', 'C', 'D'].index(pos2)
            
            p_val = pair_data['p_value']
            
            # Fill both upper and lower triangles to make the matrix symmetric
            matrix[i, j] = p_val
            matrix[j, i] = p_val
            
            is_significant = p_val < corrected_alpha
            
            # --- CORRECTED LABEL FORMATTING LOGIC ---
            # Only apply labels to the upper triangle to avoid clutter
            label_text = ""
            # Use a small epsilon for robust floating point comparison
            if p_val < 0.00005: 
                label_text = "0.0000"
            else:
                label_text = f"{p_val:.4f}"

            if is_significant:
                labels[i, j] = f"{label_text}*"
                significant_mask[i, j] = True
            else:
                labels[i, j] = label_text
        
        # --- NEW CUSTOM COLORMAP LOGIC ---
        # Define the threshold in log space
        log_alpha = np.log10(corrected_alpha) # approx -2.08
        
        # Define the colors: light red for significant, orange-to-green for non-significant
        colors = [
            (0.0, "red"),  # Values from -6 up to log_alpha will be red
            ((log_alpha - -6) / (0 - -6), "red"), # End of red section
            ((log_alpha - -6) / (0 - -6) + 1e-9, "orange"), # Start of orange section
            (1.0, "mediumseagreen") # End with green
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_map", colors)

        # Create heatmap with the custom colormap
        matrix_log = np.log10(matrix + 1e-10)
        
        sns.heatmap(matrix_log, annot=labels, fmt='', ax=ax, cmap=custom_cmap,
                    cbar_kws={'label': 'p-value (log scale)'},
                    vmin=-6, vmax=0, 
                    xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'],
                    linewidths=1.5, linecolor='white',
                    annot_kws={'size': 15, 'fontweight': 'bold'})
        
        # Add thick BLACK borders around statistically significant cells
        for i in range(4):
            for j in range(4):
                if i == j: continue # Skip diagonal
                if significant_mask[i, j] or significant_mask[j, i]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                               edgecolor='black', lw=4, zorder=10))
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name}: Pairwise p-value heatmaps (Bonferroni alpha = {corrected_alpha})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'{dataset_name}_pairwise_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   - Heatmap saved: {output_path}")


def main():
    """
    Main function to run the batch analysis.
    """
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    csv_dir = script_dir / "results" / "csv_results"
    output_dir = script_dir / "results" / "stat_results"
    
    print("=" * 80)
    print("STARTING BATCH STATISTICAL ANALYSIS")
    print(f"Input CSV folder:  {csv_dir}")
    print(f"Output text folder: {output_dir}")
    print("=" * 80)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {csv_dir}")
        return

    # Define known model prefixes to help with parsing
    model_prefixes = ['mistral', 'gemma', 'llama', 'phi', 'qwen']

    for csv_file in csv_files:
        filename = csv_file.stem.replace('_sampling_n15', '').replace('_prob', '')
        
        dataset = None
        model = None

        # Find the model name by looking for a known prefix
        for prefix in model_prefixes:
            if prefix in filename:
                parts = filename.split(prefix, 1)
                dataset = parts[0].strip('-')
                model = prefix + parts[1]
                break
        
        if not dataset or not model:
            print(f"⚠️  Skipping {csv_file.name}: cannot parse dataset and model name.")
            continue
        
        print(f"\nProcessing: {dataset} / {model}")
        
        # Analyze the file
        analysis_results = analyze_csv(csv_file)
        
        if analysis_results:
            # Format the report
            report_content = format_analysis_report(analysis_results, model, dataset)
            
            # Create dataset-specific output directory
            dataset_output_dir = output_dir / dataset
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the report
            output_filename = f"{dataset}-{model}_analysis.txt"
            output_path = dataset_output_dir / output_filename
            
            try:
                with open(output_path, 'w') as f:
                    f.write(report_content)
                print(f"   - Success: Saved analysis to {output_path}")
            except Exception as e:
                print(f"   - Failure: Could not write to {output_path}: {e}")

    # After processing all CSVs, organize results by dataset for heatmap generation
    dataset_results = {}
    
    for csv_file in csv_files:
        filename = csv_file.stem.replace('_sampling_n15', '').replace('_prob', '')
        
        dataset = None
        model = None

        # Find the model name by looking for a known prefix
        for prefix in model_prefixes:
            if prefix in filename:
                parts = filename.split(prefix, 1)
                dataset = parts[0].strip('-')
                model = prefix + parts[1]
                break
        
        if not dataset or not model:
            continue
        
        print(f"\nProcessing: {dataset} / {model}")
        
        # Analyze the file
        analysis_results = analyze_csv(csv_file)
        
        if analysis_results:
            # Store results for heatmap generation
            if dataset not in dataset_results:
                dataset_results[dataset] = {}
            dataset_results[dataset][model] = analysis_results
    
    # Generate heatmaps for each dataset
    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS")
    print("=" * 80)
    
    for dataset, models_data in dataset_results.items():
        print(f"\nCreating heatmap for: {dataset}")
        dataset_output_dir = output_dir / dataset
        create_pairwise_heatmap(dataset, models_data, dataset_output_dir)
    
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
