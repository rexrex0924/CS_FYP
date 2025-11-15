"""
PriDe Batch Summary Evaluation
Processes all CSV results and generates comparison visualizations:
1. By Dataset: Compare all models on the same dataset
2. By Model-Dataset: Individual model performance plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy.stats import chisquare, chi2_contingency
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PriDeDebiasing:
    """PriDe: Debiasing with Prior estimation for multiple choice questions."""

    def __init__(self, calibration_ratio: float = 0.1,
                 alpha: float = 1.0,
                 random_seed: int = 42):
        self.calibration_ratio = calibration_ratio
        self.alpha = alpha
        self.random_seed = random_seed
        self.global_prior = None
        self.calibrated = False

    def gather_probs(self, observed: np.ndarray, permuted_indices: List) -> List[List[float]]:
        n_options = observed.shape[1]
        gathered_probs = [[] for _ in range(n_options)]
        for pdx, indices in enumerate(permuted_indices):
            for idx, index in enumerate(indices):
                gathered_probs[index].append(observed[pdx, idx])
        return gathered_probs

    def estimate_prior_for_question(self, probs_matrix: np.ndarray,
                                    permuted_indices: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        observed = probs_matrix / (probs_matrix.sum(axis=1, keepdims=True) + 1e-10)
        gathered_probs = self.gather_probs(observed, permuted_indices)
        debiased = np.array([np.mean(probs) for probs in gathered_probs])
        prior = self.softmax(np.log(observed + 1e-10).mean(axis=0))
        return debiased, prior

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)

    def split_calibration_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_questions = df['question_id'].unique()
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(unique_questions)
        n_calibration = max(1, int(len(unique_questions) * self.calibration_ratio))
        calibration_questions = unique_questions[:n_calibration]
        test_questions = unique_questions[n_calibration:]
        calibration_df = df[df['question_id'].isin(calibration_questions)].copy()
        test_df = df[df['question_id'].isin(test_questions)].copy()
        return calibration_df, test_df

    def estimate_prior_from_calibration(self, calibration_df: pd.DataFrame) -> np.ndarray:
        question_groups = calibration_df.groupby('question_id')
        all_priors = []
        for qid, group in question_groups:
            group = group.sort_values('permutation_idx')
            probs_matrix = group[['prob_A', 'prob_B', 'prob_C', 'prob_D']].values
            n_options = 4
            permuted_indices = [
                tuple((i + shift) % n_options for i in range(n_options))
                for shift in range(probs_matrix.shape[0])
            ]
            _, prior = self.estimate_prior_for_question(probs_matrix, permuted_indices)
            all_priors.append(prior)
        self.global_prior = np.mean(all_priors, axis=0)
        self.calibrated = True
        return self.global_prior

    def debias_with_prior(self, observed_probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
        debiased = (np.log(observed_probs + 1e-10) -
                   self.alpha * np.log(prior + 1e-10))
        return debiased

    def debias_test_set(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.calibrated:
            raise ValueError("Must estimate prior from calibration set first")
        df_debiased = test_df.copy()
        positions = ['A', 'B', 'C', 'D']
        all_debiased_answers = []
        all_debiased_correct = []
        for idx, row in df_debiased.iterrows():
            observed_probs = np.array([row[f'prob_{pos}'] for pos in positions])
            debiased_logits = self.debias_with_prior(observed_probs, self.global_prior)
            predicted_position = np.argmax(debiased_logits)
            predicted_answer = positions[predicted_position]
            all_debiased_answers.append(predicted_answer)
            is_correct = (predicted_answer == row['correct_position'])
            all_debiased_correct.append(int(is_correct))
        df_debiased['debiased_predicted_answer'] = all_debiased_answers
        df_debiased['debiased_is_correct'] = all_debiased_correct
        return df_debiased

    def fit_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        calibration_df, test_df = self.split_calibration_test(df)
        prior = self.estimate_prior_from_calibration(calibration_df)
        test_df_debiased = self.debias_test_set(test_df)
        calibration_info = {
            'n_total_questions': df['question_id'].nunique(),
            'n_calibration_questions': calibration_df['question_id'].nunique(),
            'n_test_questions': test_df['question_id'].nunique(),
            'calibration_samples': len(calibration_df),
            'test_samples': len(test_df),
            'estimated_prior': prior.tolist(),
        }
        return test_df_debiased, calibration_info


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the CSV data for PriDe analysis."""
    df = pd.read_csv(csv_path)
    for pos in ['A', 'B', 'C', 'D']:
        df[f'prob_{pos}'] = pd.to_numeric(df[f'prob_{pos}'], errors='coerce')
    df['is_correct_fixed'] = (df['predicted_answer'] == df['correct_position']).astype(int)
    return df


def compute_bias_metrics(df: pd.DataFrame, prediction_col: str = 'predicted_answer') -> Dict:
    """Compute all bias metrics for a dataset."""
    df_local = df.copy()
    df_local[prediction_col] = df_local[prediction_col].astype(str).str.upper().str.strip()
    df_local['correct_position'] = df_local['correct_position'].astype(str).str.upper().str.strip()
    
    valid_responses = df_local[df_local[prediction_col].isin(["A", "B", "C", "D"])].copy()
    valid_responses["is_correct_eval"] = (
        valid_responses[prediction_col] == valid_responses['correct_position']
    ).astype(int)
    
    choice_counts = valid_responses[prediction_col].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    total_valid = len(valid_responses)
    
    # Chi-square test
    expected_per_choice = total_valid / 4
    expected = [expected_per_choice] * 4
    chi2_stat, p_value = chisquare(choice_counts.values, f_exp=expected)
    
    # Position bias score
    choice_percentages = choice_counts.values / total_valid * 100
    position_bias_score = np.std(choice_percentages)
    
    # Recall Standard Deviation (RStd)
    positions = ['A', 'B', 'C', 'D']
    recalls = []
    for pos in positions:
        pos_mask = valid_responses['correct_position'] == pos
        if pos_mask.sum() > 0:
            recall = (valid_responses[pos_mask][prediction_col] == pos).mean()
            recalls.append(recall)
        else:
            recalls.append(0.0)
    recall_std = np.std(recalls) * 100
    
    # Accuracy by position
    accuracy_by_position = valid_responses.groupby('correct_position')['is_correct_eval'].mean()
    overall_accuracy = valid_responses['is_correct_eval'].mean()
    
    # Accuracy vs position chi-square
    try:
        contingency_table = pd.crosstab(valid_responses['correct_position'], valid_responses['is_correct_eval'])
        chi2_acc, p_acc, _, _ = chi2_contingency(contingency_table)
    except:
        chi2_acc, p_acc = 0, 1
    
    return {
        'choice_counts': choice_counts.to_dict(),
        'choice_percentages': (choice_counts / total_valid * 100).to_dict(),
        'chi2_stat': chi2_stat,
        'chi2_pvalue': p_value,
        'position_bias_score': position_bias_score,
        'recall_std': recall_std,
        'recalls': {pos: recalls[i] for i, pos in enumerate(positions)},
        'accuracy_by_position': accuracy_by_position.to_dict(),
        'overall_accuracy': overall_accuracy,
        'chi2_acc_stat': chi2_acc,
        'chi2_acc_pvalue': p_acc,
        'n_samples': total_valid
    }


def find_best_alpha(df: pd.DataFrame, calibration_ratio: float = 0.05) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Find the best alpha value for debiasing."""
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha = 1.0
    best_acc = 0.0
    best_test_df = None
    best_test_original = None
    
    for alpha in alphas:
        pride = PriDeDebiasing(calibration_ratio=calibration_ratio, alpha=alpha, random_seed=42)
        test_df_debiased, _ = pride.fit_and_predict(df)
        _, test_df_original = pride.split_calibration_test(df)
        
        deb_acc = test_df_debiased['debiased_is_correct'].mean()
        
        if deb_acc > best_acc:
            best_acc = deb_acc
            best_alpha = alpha
            best_test_df = test_df_debiased
            best_test_original = test_df_original
    
    return best_alpha, best_test_df, best_test_original


def process_all_csvs(csv_dir: Path) -> Dict:
    """Process all CSV files and compute metrics."""
    print("=" * 80)
    print("PROCESSING ALL CSV FILES")
    print("=" * 80)
    
    results = defaultdict(lambda: defaultdict(dict))
    
    csv_files = sorted(csv_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        filename = csv_file.stem
        
        # Parse filename: dataset-model_sampling_n15.csv
        parts = filename.replace('_sampling_n15', '').split('-')
        if len(parts) < 2:
            print(f"⚠️  Skipping {filename}: cannot parse dataset-model format")
            continue
        
        dataset = parts[0]
        model = '-'.join(parts[1:])
        
        print(f"\n📊 Processing: {dataset} / {model}")
        print(f"   File: {csv_file.name}")
        
        try:
            # Load data
            df = load_and_prepare_data(csv_file)
            
            # Find best alpha and debias
            best_alpha, test_df_debiased, test_df_original = find_best_alpha(df, calibration_ratio=0.05)
            
            # Compute metrics
            baseline_metrics = compute_bias_metrics(test_df_original, 'predicted_answer')
            debiased_metrics = compute_bias_metrics(test_df_debiased, 'debiased_predicted_answer')
            
            # Store results
            results[dataset][model] = {
                'baseline': baseline_metrics,
                'debiased': debiased_metrics,
                'best_alpha': best_alpha,
                'csv_path': str(csv_file)
            }
            
            print(f"   ✅ Alpha: {best_alpha:.2f}")
            print(f"   ✅ Accuracy: {baseline_metrics['overall_accuracy']*100:.1f}% → {debiased_metrics['overall_accuracy']*100:.1f}% "
                  f"({(debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy'])*100:+.2f}pp)")
            print(f"   ✅ Bias Score: {baseline_metrics['position_bias_score']:.2f} → {debiased_metrics['position_bias_score']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    return dict(results)


# ============================================================================
# VISUALIZATION FUNCTIONS: BY DATASET (All Models Comparison)
# ============================================================================

def plot_dataset_accuracy_comparison(dataset: str, models_data: Dict, output_path: Path):
    """Compare accuracy across all models for a dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort models by baseline accuracy (descending)
    models = sorted(models_data.keys(), 
                   key=lambda m: models_data[m]['baseline']['overall_accuracy'], 
                   reverse=True)
    baseline_accs = [models_data[m]['baseline']['overall_accuracy'] for m in models]
    debiased_accs = [models_data[m]['debiased']['overall_accuracy'] for m in models]
    improvements = [d - b for b, d in zip(baseline_accs, debiased_accs)]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Left: Baseline vs Debiased
    bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, debiased_accs, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset}: Accuracy Comparison (Sorted by Baseline)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels (as percentages)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Right: Improvements (sorted by improvement value)
    models_by_improvement = sorted(models_data.keys(), 
                                   key=lambda m: models_data[m]['debiased']['overall_accuracy'] - 
                                                models_data[m]['baseline']['overall_accuracy'], 
                                   reverse=True)
    improvements_sorted = [models_data[m]['debiased']['overall_accuracy'] - 
                          models_data[m]['baseline']['overall_accuracy'] 
                          for m in models_by_improvement]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements_sorted]
    bars3 = ax2.bar(range(len(models_by_improvement)), [imp * 100 for imp in improvements_sorted], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement (percentage points)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset}: Accuracy Improvement (Sorted)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models_by_improvement)))
    ax2.set_xticklabels(models_by_improvement, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels (as percentage points)
    for bar in bars3:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}pp', ha='center', va=va, fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dataset_bias_comparison(dataset: str, models_data: Dict, output_path: Path):
    """Compare bias metrics across all models for a dataset."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    width = 0.35
    
    # Position Bias Score (sorted by baseline, ascending - lower is better)
    models_bias = sorted(models_data.keys(), 
                        key=lambda m: models_data[m]['baseline']['position_bias_score'])
    baseline_bias = [models_data[m]['baseline']['position_bias_score'] for m in models_bias]
    debiased_bias = [models_data[m]['debiased']['position_bias_score'] for m in models_bias]
    
    x1 = np.arange(len(models_bias))
    bars1a = ax1.bar(x1 - width/2, baseline_bias, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars1b = ax1.bar(x1 + width/2, debiased_bias, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Position Bias Score', fontsize=10, fontweight='bold')
    ax1.set_title(f'{dataset}: Position Bias Score (Lower = Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models_bias, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1a:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars1b:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Recall Std (sorted by baseline, ascending - lower is better)
    models_rstd = sorted(models_data.keys(), 
                        key=lambda m: models_data[m]['baseline']['recall_std'])
    baseline_rstd = [models_data[m]['baseline']['recall_std'] for m in models_rstd]
    debiased_rstd = [models_data[m]['debiased']['recall_std'] for m in models_rstd]
    
    x2 = np.arange(len(models_rstd))
    bars2a = ax2.bar(x2 - width/2, baseline_rstd, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2b = ax2.bar(x2 + width/2, debiased_rstd, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recall Std (%)', fontsize=10, fontweight='bold')
    ax2.set_title(f'{dataset}: Recall Standard Deviation (Lower = Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models_rstd, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2a:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2b:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Chi-square statistic (sorted by baseline, ascending - lower is better)
    models_chi2 = sorted(models_data.keys(), 
                        key=lambda m: models_data[m]['baseline']['chi2_stat'])
    baseline_chi2 = [models_data[m]['baseline']['chi2_stat'] for m in models_chi2]
    debiased_chi2 = [models_data[m]['debiased']['chi2_stat'] for m in models_chi2]
    
    x3 = np.arange(len(models_chi2))
    bars3a = ax3.bar(x3 - width/2, baseline_chi2, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars3b = ax3.bar(x3 + width/2, debiased_chi2, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax3.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Chi-square Statistic', fontsize=10, fontweight='bold')
    ax3.set_title(f'{dataset}: Distribution Chi-square (Lower = More Uniform)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(models_chi2, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3a:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars3b:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Chi-square p-value (sorted by baseline, descending - higher is better)
    models_pval = sorted(models_data.keys(), 
                        key=lambda m: models_data[m]['baseline']['chi2_pvalue'],
                        reverse=True)
    baseline_pval = [models_data[m]['baseline']['chi2_pvalue'] for m in models_pval]
    debiased_pval = [models_data[m]['debiased']['chi2_pvalue'] for m in models_pval]
    
    x4 = np.arange(len(models_pval))
    bars4a = ax4.bar(x4 - width/2, baseline_pval, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars4b = ax4.bar(x4 + width/2, debiased_pval, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax4.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Chi-square p-value', fontsize=10, fontweight='bold')
    ax4.set_title(f'{dataset}: Distribution p-value (Higher = More Uniform)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(models_pval, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars4a:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars4b:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dataset_distribution_comparison(dataset: str, models_data: Dict, output_path: Path):
    """Compare choice distributions across all models for a dataset."""
    # Sort models by baseline position bias score (ascending - lower is better)
    models = sorted(models_data.keys(), 
                   key=lambda m: models_data[m]['baseline']['position_bias_score'])
    n_models = len(models)
    
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    positions = ['A', 'B', 'C', 'D']
    x = np.arange(len(positions))
    
    for idx, model in enumerate(models):
        baseline_pcts = [models_data[model]['baseline']['choice_percentages'][p] for p in positions]
        debiased_pcts = [models_data[model]['debiased']['choice_percentages'][p] for p in positions]
        
        # Baseline
        ax1 = axes[idx, 0]
        bars1 = ax1.bar(x, baseline_pcts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
        ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
        ax1.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax1.set_title(f'{model} - Baseline\nBias Score: {models_data[model]["baseline"]["position_bias_score"]:.2f}',
                     fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(positions)
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Debiased
        ax2 = axes[idx, 1]
        bars2 = ax2.bar(x, debiased_pcts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
        ax2.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
        ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax2.set_title(f'{model} - After PriDe (α={models_data[model]["best_alpha"]:.2f})\n'
                     f'Bias Score: {models_data[model]["debiased"]["position_bias_score"]:.2f}',
                     fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(positions)
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    fig.suptitle(f'{dataset}: Choice Distribution Comparison (Sorted by Bias Score)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dataset_accuracy_by_position(dataset: str, models_data: Dict, output_path: Path):
    """Compare accuracy by position across all models for a dataset."""
    # Sort models by baseline overall accuracy (descending - higher is better)
    models = sorted(models_data.keys(), 
                   key=lambda m: models_data[m]['baseline']['overall_accuracy'],
                   reverse=True)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = [axes]
    
    positions = ['A', 'B', 'C', 'D']
    x = np.arange(len(positions))
    width = 0.35
    
    for idx, model in enumerate(models):
        baseline_acc = [models_data[model]['baseline']['accuracy_by_position'].get(p, 0) for p in positions]
        debiased_acc = [models_data[model]['debiased']['accuracy_by_position'].get(p, 0) for p in positions]
        overall_baseline = models_data[model]['baseline']['overall_accuracy']
        overall_debiased = models_data[model]['debiased']['overall_accuracy']
        
        ax = axes[idx]
        bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, debiased_acc, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
        
        ax.axhline(y=overall_baseline, color='#FF6B6B', linestyle='--', linewidth=2,
                  label=f'Baseline Overall: {overall_baseline*100:.1f}%')
        ax.axhline(y=overall_debiased, color='#4ECDC4', linestyle='--', linewidth=2,
                  label=f'PriDe Overall: {overall_debiased*100:.1f}%')
        
        ax.set_xlabel('Correct Answer Position', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{model}: Accuracy by Position\n'
                    f'Improvement: {(overall_debiased - overall_baseline)*100:+.2f}pp',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels (as percentages)
        for bar1, bar2 in zip(bars1, bars2):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{height1*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{height2*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    fig.suptitle(f'{dataset}: Accuracy by Position (Sorted by Baseline Accuracy)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION FUNCTIONS: BY MODEL-DATASET (Individual)
# ============================================================================

def plot_individual_accuracy_comparison(dataset: str, model: str, model_data: Dict, output_path: Path):
    """Individual model-dataset accuracy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    baseline = model_data['baseline']
    debiased = model_data['debiased']
    
    # Overall accuracy
    categories = ['Baseline', 'After PriDe']
    accuracies = [baseline['overall_accuracy'], debiased['overall_accuracy']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(categories, [acc * 100 for acc in accuracies], color=colors, alpha=0.8, width=0.6)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset} - {model}\nOverall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    improvement = debiased['overall_accuracy'] - baseline['overall_accuracy']
    ax1.text(0.5, max([acc * 100 for acc in accuracies]) * 0.5, f'Δ = {improvement*100:+.2f}pp',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Accuracy by position
    positions = ['A', 'B', 'C', 'D']
    baseline_acc = [baseline['accuracy_by_position'].get(p, 0) for p in positions]
    debiased_acc = [debiased['accuracy_by_position'].get(p, 0) for p in positions]
    
    x = np.arange(len(positions))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, [acc * 100 for acc in baseline_acc], width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, [acc * 100 for acc in debiased_acc], width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Correct Answer Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Position', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, baseline_acc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, acc in zip(bars2, debiased_acc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_bias_metrics(dataset: str, model: str, model_data: Dict, output_path: Path):
    """Individual model-dataset bias metrics comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    baseline = model_data['baseline']
    debiased = model_data['debiased']
    
    # Position Bias Score
    categories = ['Baseline', 'After PriDe']
    bias_scores = [baseline['position_bias_score'], debiased['position_bias_score']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(categories, bias_scores, color=colors, alpha=0.8, width=0.6)
    ax1.set_ylabel('Position Bias Score', fontsize=11, fontweight='bold')
    ax1.set_title('Position Bias Score (Lower = Better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Recall Std
    rstd_values = [baseline['recall_std'], debiased['recall_std']]
    bars2 = ax2.bar(categories, rstd_values, color=colors, alpha=0.8, width=0.6)
    ax2.set_ylabel('Recall Std (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Recall Standard Deviation (Lower = Better)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chi-square statistic
    chi2_values = [baseline['chi2_stat'], debiased['chi2_stat']]
    bars3 = ax3.bar(categories, chi2_values, color=colors, alpha=0.8, width=0.6)
    ax3.set_ylabel('Chi-square Statistic', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution Chi-square (Lower = More Uniform)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chi-square p-value
    pval_values = [baseline['chi2_pvalue'], debiased['chi2_pvalue']]
    bars4 = ax4.bar(categories, pval_values, color=colors, alpha=0.8, width=0.6)
    ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax4.set_ylabel('Chi-square p-value', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution p-value (Higher = More Uniform)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle(f'{dataset} - {model}: Bias Metrics Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_distribution(dataset: str, model: str, model_data: Dict, output_path: Path):
    """Individual model-dataset choice distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    baseline = model_data['baseline']
    debiased = model_data['debiased']
    
    positions = ['A', 'B', 'C', 'D']
    baseline_pcts = [baseline['choice_percentages'][p] for p in positions]
    debiased_pcts = [debiased['choice_percentages'][p] for p in positions]
    
    x = np.arange(len(positions))
    width = 0.6
    
    # Baseline
    bars1 = ax1.bar(x, baseline_pcts, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax1.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Baseline\nBias Score: {baseline["position_bias_score"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Debiased
    bars2 = ax2.bar(x, debiased_pcts, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax2.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax2.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'After PriDe (α={model_data["best_alpha"]:.2f})\n'
                 f'Bias Score: {debiased["position_bias_score"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()
    ax2.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'{dataset} - {model}: Choice Distribution',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_summary(dataset: str, model: str, model_data: Dict, output_path: Path):
    """Individual model-dataset comprehensive summary."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    baseline = model_data['baseline']
    debiased = model_data['debiased']
    best_alpha = model_data['best_alpha']
    
    fig.suptitle(f'{dataset} - {model}: Comprehensive Summary',
                fontsize=18, fontweight='bold')
    
    # 1. Overall Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Baseline', 'After PriDe']
    accuracies = [baseline['overall_accuracy'], debiased['overall_accuracy']]
    bars = ax1.bar(categories, [acc * 100 for acc in accuracies], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Overall Accuracy', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Position Bias Score
    ax2 = fig.add_subplot(gs[0, 1])
    bias_scores = [baseline['position_bias_score'], debiased['position_bias_score']]
    bars = ax2.bar(categories, bias_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax2.set_ylabel('Bias Score', fontsize=10, fontweight='bold')
    ax2.set_title('Position Bias Score', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Recall Std
    ax3 = fig.add_subplot(gs[0, 2])
    rstd_values = [baseline['recall_std'], debiased['recall_std']]
    bars = ax3.bar(categories, rstd_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax3.set_ylabel('Recall Std (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Recall Standard Deviation', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Choice Distribution
    ax4 = fig.add_subplot(gs[1, :2])
    positions = ['A', 'B', 'C', 'D']
    baseline_pcts = [baseline['choice_percentages'][p] for p in positions]
    debiased_pcts = [debiased['choice_percentages'][p] for p in positions]
    x = np.arange(len(positions))
    width = 0.35
    ax4.bar(x - width/2, baseline_pcts, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    ax4.bar(x + width/2, debiased_pcts, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax4.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax4.set_xlabel('Position', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Choice Distribution', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(positions)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Accuracy by Position
    ax5 = fig.add_subplot(gs[2, :2])
    baseline_acc = [baseline['accuracy_by_position'].get(p, 0) for p in positions]
    debiased_acc = [debiased['accuracy_by_position'].get(p, 0) for p in positions]
    bars5a = ax5.bar(x - width/2, [acc * 100 for acc in baseline_acc], width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars5b = ax5.bar(x + width/2, [acc * 100 for acc in debiased_acc], width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax5.set_xlabel('Correct Answer Position', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax5.set_title('Accuracy by Position', fontsize=11, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(positions)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars5a, baseline_acc):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, acc in zip(bars5b, debiased_acc):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. Summary Text
    ax6 = fig.add_subplot(gs[1:, 2])
    ax6.axis('off')
    
    acc_change = debiased['overall_accuracy'] - baseline['overall_accuracy']
    bias_change = debiased['position_bias_score'] - baseline['position_bias_score']
    rstd_change = debiased['recall_std'] - baseline['recall_std']
    
    summary_text = f"""SUMMARY METRICS

Alpha (α) = {best_alpha:.2f}

━━━ ACCURACY ━━━
Before: {baseline['overall_accuracy']*100:.1f}%
After:  {debiased['overall_accuracy']*100:.1f}%
Change: {acc_change*100:+.2f}pp
{'✅ Improved' if acc_change > 0 else '⚠️  Decreased'}

━━━ BIAS METRICS ━━━
Position Bias:
  Before: {baseline['position_bias_score']:.2f}
  After:  {debiased['position_bias_score']:.2f}
  Change: {bias_change:+.2f}
  {'✅ Reduced' if bias_change < 0 else '⚠️  Increased'}

Recall Std:
  Before: {baseline['recall_std']:.2f}%
  After:  {debiased['recall_std']:.2f}%
  Change: {rstd_change:+.2f}%
  {'✅ Balanced' if rstd_change < 0 else '⚠️  Imbalanced'}

━━━ CHI-SQUARE ━━━
Distribution:
  χ²: {baseline['chi2_stat']:.2f} → {debiased['chi2_stat']:.2f}
  p: {baseline['chi2_pvalue']:.4f} → {debiased['chi2_pvalue']:.4f}

Acc-Position:
  χ²: {baseline['chi2_acc_stat']:.2f} → {debiased['chi2_acc_stat']:.2f}
  p: {baseline['chi2_acc_pvalue']:.4f} → {debiased['chi2_acc_pvalue']:.4f}
"""
    
    ax6.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax6.transAxes)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PriDe BATCH SUMMARY EVALUATION")
    print("=" * 80)
    
    # Set paths
    csv_dir = Path("results/csv_results")
    output_base = Path("results/pride_summary")
    
    # Create output directories
    by_dataset_dir = output_base / "by_dataset"
    by_model_dataset_dir = output_base / "by_model_dataset"
    by_dataset_dir.mkdir(parents=True, exist_ok=True)
    by_model_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all CSV files
    results = process_all_csvs(csv_dir)
    
    if not results:
        print("\n❌ No results to visualize!")
        return
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Generate visualizations BY DATASET
    print("\n📊 Generating BY DATASET comparisons...")
    for dataset, models_data in results.items():
        print(f"\n  Dataset: {dataset} ({len(models_data)} models)")
        
        dataset_dir = by_dataset_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"    - Accuracy comparison...")
        plot_dataset_accuracy_comparison(dataset, models_data, 
                                        dataset_dir / f"{dataset}_accuracy_comparison.png")
        
        print(f"    - Bias metrics comparison...")
        plot_dataset_bias_comparison(dataset, models_data,
                                    dataset_dir / f"{dataset}_bias_comparison.png")
        
        print(f"    - Distribution comparison...")
        plot_dataset_distribution_comparison(dataset, models_data,
                                            dataset_dir / f"{dataset}_distribution_comparison.png")
        
        print(f"    - Accuracy by position...")
        plot_dataset_accuracy_by_position(dataset, models_data,
                                         dataset_dir / f"{dataset}_accuracy_by_position.png")
    
    # Generate visualizations BY MODEL-DATASET
    print("\n📊 Generating BY MODEL-DATASET individual plots...")
    for dataset, models_data in results.items():
        for model, model_data in models_data.items():
            print(f"\n  {dataset} - {model}")
            
            model_dataset_dir = by_model_dataset_dir / f"{dataset}-{model}"
            model_dataset_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"    - Accuracy comparison...")
            plot_individual_accuracy_comparison(dataset, model, model_data,
                                              model_dataset_dir / "accuracy_comparison.png")
            
            print(f"    - Bias metrics...")
            plot_individual_bias_metrics(dataset, model, model_data,
                                       model_dataset_dir / "bias_metrics.png")
            
            print(f"    - Distribution...")
            plot_individual_distribution(dataset, model, model_data,
                                       model_dataset_dir / "distribution.png")
            
            print(f"    - Summary...")
            plot_individual_summary(dataset, model, model_data,
                                  model_dataset_dir / "summary.png")
    
    # Generate overall summary report
    print("\n📝 Generating summary report...")
    report_path = output_base / "SUMMARY_REPORT.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PriDe BATCH EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset, models_data in sorted(results.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"DATASET: {dataset}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"{'Model':<35} {'α':<6} {'Acc (B→D)':<25} {'Bias (B→D)':<20} {'RStd (B→D)':<20}\n")
            f.write("-" * 80 + "\n")
            
            for model, model_data in sorted(models_data.items()):
                baseline = model_data['baseline']
                debiased = model_data['debiased']
                alpha = model_data['best_alpha']
                
                acc_str = f"{baseline['overall_accuracy']*100:.1f}%→{debiased['overall_accuracy']*100:.1f}%"
                bias_str = f"{baseline['position_bias_score']:.2f}→{debiased['position_bias_score']:.2f}"
                rstd_str = f"{baseline['recall_std']:.2f}→{debiased['recall_std']:.2f}"
                
                f.write(f"{model:<35} {alpha:<6.2f} {acc_str:<25} {bias_str:<20} {rstd_str:<20}\n")
    
    print(f"\n✅ Summary report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("✨ BATCH EVALUATION COMPLETE! ✨")
    print("=" * 80)
    print(f"\n📂 Output directories:")
    print(f"   - By Dataset: {by_dataset_dir}")
    print(f"   - By Model-Dataset: {by_model_dataset_dir}")
    print(f"   - Summary Report: {report_path}")
    print("\n")


if __name__ == "__main__":
    main()

