import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
import argparse
from scipy.stats import chisquare, chi2_contingency
warnings.filterwarnings('ignore')


class PriDeDebiasing:
    """
    PriDe: Debiasing with Prior estimation for multiple choice questions.

    Based on the official implementation from:
    https://github.com/chujiezheng/LLM-MCQ-Bias

    Key principle:
    1. Use a SMALL SUBSET (calibration set) to estimate positional bias
    2. Apply the estimated bias to the REST of the data (test set)
    3. Adjustable debiasing strength (alpha) for optimal results
    """

    def __init__(self, calibration_ratio: float = 0.05,
                 alpha: float = 1.0,
                 random_seed: int = 42):
        """
        Args:
            calibration_ratio: Fraction of data for estimating prior (default 5%)
            alpha: Debiasing strength (0=none, 1=full, 0.5=half)
            random_seed: Random seed for splitting data
        """
        self.calibration_ratio = calibration_ratio
        self.alpha = alpha
        self.random_seed = random_seed
        self.global_prior = None
        self.calibrated = False

    def gather_probs(self, observed: np.ndarray, permuted_indices: List) -> List[List[float]]:
        """
        Gather probabilities assigned to each answer content across permutations.

        Args:
            observed: Shape (n_permutations, n_options) probability matrix
            permuted_indices: List of permutation tuples indicating option order

        Returns:
            List of probabilities for each original answer option
        """
        n_options = observed.shape[1]
        gathered_probs = [[] for _ in range(n_options)]

        for pdx, indices in enumerate(permuted_indices):
            for idx, index in enumerate(indices):
                gathered_probs[index].append(observed[pdx, idx])

        return gathered_probs

    def estimate_prior_for_question(self,
                                    probs_matrix: np.ndarray,
                                    permuted_indices: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prior and debiased probabilities for a single question.

        Args:
            probs_matrix: Shape (n_permutations, n_options) - probabilities for each permutation
            permuted_indices: List of tuples showing how options are permuted

        Returns:
            debiased: Debiased probabilities for each original answer option
            prior: Prior probability for each position in this question
        """
        # Normalize probabilities
        observed = probs_matrix / (probs_matrix.sum(axis=1, keepdims=True) + 1e-10)

        # Gather probabilities assigned to each answer content across permutations
        gathered_probs = self.gather_probs(observed, permuted_indices)

        # Average probabilities for each answer content (debiased)
        debiased = np.array([np.mean(probs) for probs in gathered_probs])

        # Estimate prior: average log probabilities across permutations, then softmax
        prior = self.softmax(np.log(observed + 1e-10).mean(axis=0))

        return debiased, prior

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)

    def split_calibration_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into calibration set (for estimating prior) and test set.

        Args:
            df: Full dataset

        Returns:
            calibration_df: Subset for estimating prior
            test_df: Subset for evaluation
        """
        # Get unique question IDs
        unique_questions = df['question_id'].unique()

        # Shuffle questions
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(unique_questions)

        # Split questions
        n_calibration = max(1, int(len(unique_questions) * self.calibration_ratio))
        calibration_questions = unique_questions[:n_calibration]
        test_questions = unique_questions[n_calibration:]

        # Split dataframe
        calibration_df = df[df['question_id'].isin(calibration_questions)].copy()
        test_df = df[df['question_id'].isin(test_questions)].copy()

        return calibration_df, test_df

    def estimate_prior_from_calibration(self, calibration_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate global prior from calibration set only.

        Args:
            calibration_df: Calibration subset of data

        Returns:
            Global prior probabilities for positions
        """
        question_groups = calibration_df.groupby('question_id')
        all_priors = []

        for qid, group in question_groups:
            group = group.sort_values('permutation_idx')

            # Get probability matrix
            probs_matrix = group[['prob_A', 'prob_B', 'prob_C', 'prob_D']].values

            # Infer permutation indices (cyclic)
            n_options = 4
            permuted_indices = [
                tuple((i + shift) % n_options for i in range(n_options))
                for shift in range(probs_matrix.shape[0])
            ]

            # Estimate prior for this question
            _, prior = self.estimate_prior_for_question(probs_matrix, permuted_indices)
            all_priors.append(prior)

        # Calculate global prior (average across calibration questions)
        self.global_prior = np.mean(all_priors, axis=0)
        self.calibrated = True

        return self.global_prior

    def debias_with_prior(self, observed_probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
        """
        Apply PriDe debiasing formula with adjustable strength.

        Formula: debiased = log(observed) - alpha * log(prior)
        - alpha=0: No debiasing (keeps original)
        - alpha=1: Full debiasing (original PriDe)
        - alpha=0.5: Partial debiasing (compromise)

        Args:
            observed_probs: Original probabilities from the model
            prior: Estimated prior probabilities for positions

        Returns:
            Debiased logits
        """
        debiased = (np.log(observed_probs + 1e-10) -
                   self.alpha * np.log(prior + 1e-10))
        return debiased

    def debias_test_set(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply debiasing to test set using the estimated prior.

        Args:
            test_df: Test subset of data

        Returns:
            DataFrame with debiased predictions
        """
        if not self.calibrated:
            raise ValueError("Must estimate prior from calibration set first")

        df_debiased = test_df.copy()
        positions = ['A', 'B', 'C', 'D']

        all_debiased_answers = []
        all_debiased_correct = []

        for idx, row in df_debiased.iterrows():
            # Get observed probabilities for this permutation
            observed_probs = np.array([row[f'prob_{pos}'] for pos in positions])

            # Apply debiasing
            debiased_logits = self.debias_with_prior(observed_probs, self.global_prior)

            # Get prediction (argmax in debiased space)
            predicted_position = np.argmax(debiased_logits)
            predicted_answer = positions[predicted_position]

            all_debiased_answers.append(predicted_answer)

            # Check correctness
            is_correct = (predicted_answer == row['correct_position'])
            all_debiased_correct.append(int(is_correct))

        df_debiased['debiased_predicted_answer'] = all_debiased_answers
        df_debiased['debiased_is_correct'] = all_debiased_correct

        return df_debiased

    def fit_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline: split data, estimate prior, debias test set.

        Args:
            df: Full dataset

        Returns:
            test_df_debiased: Debiased test set
            calibration_info: Information about calibration
        """
        # Split data
        calibration_df, test_df = self.split_calibration_test(df)

        # Estimate prior from calibration set
        prior = self.estimate_prior_from_calibration(calibration_df)

        # Debias test set
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

    def evaluate_debiasing(self, test_df_original: pd.DataFrame,
                          test_df_debiased: pd.DataFrame) -> Dict:
        """
        Evaluate the effect of debiasing on test set.

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        results = {}

        # Fix original accuracy calculation
        test_df_original = test_df_original.copy()
        test_df_original['is_correct_fixed'] = (
            test_df_original['predicted_answer'] == test_df_original['correct_position']
        ).astype(int)

        # Overall accuracy
        results['original_accuracy'] = test_df_original['is_correct_fixed'].mean()
        results['debiased_accuracy'] = test_df_debiased['debiased_is_correct'].mean()
        results['accuracy_improvement'] = (
            results['debiased_accuracy'] - results['original_accuracy']
        )

        # Per-question accuracy
        orig_per_q = test_df_original.groupby('question_id')['is_correct_fixed'].mean()
        deb_per_q = test_df_debiased.groupby('question_id')['debiased_is_correct'].mean()

        results['original_per_question_accuracy'] = orig_per_q.mean()
        results['debiased_per_question_accuracy'] = deb_per_q.mean()

        # Position distribution
        orig_dist = test_df_original['predicted_answer'].value_counts(normalize=True)
        deb_dist = test_df_debiased['debiased_predicted_answer'].value_counts(normalize=True)

        results['position_distribution'] = {
            'original': orig_dist.to_dict(),
            'debiased': deb_dist.to_dict()
        }

        # Uniformity metrics
        results['original_position_variance'] = orig_dist.var()
        results['debiased_position_variance'] = deb_dist.var()
        results['uniformity_improvement'] = (
            results['original_position_variance'] - results['debiased_position_variance']
        )

        # Consistency across permutations
        orig_consistency = test_df_original.groupby('question_id')['predicted_answer'].apply(
            lambda x: (x == x.mode()[0] if len(x.mode()) > 0 else False).mean()
        ).mean()

        deb_consistency = test_df_debiased.groupby('question_id')['debiased_predicted_answer'].apply(
            lambda x: (x == x.mode()[0] if len(x.mode()) > 0 else False).mean()
        ).mean()

        results['original_consistency'] = orig_consistency
        results['debiased_consistency'] = deb_consistency
        results['consistency_improvement'] = deb_consistency - orig_consistency

        return results


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the CSV data for PriDe analysis."""
    df = pd.read_csv(csv_path)

    # Ensure probability columns are numeric
    for pos in ['A', 'B', 'C', 'D']:
        df[f'prob_{pos}'] = pd.to_numeric(df[f'prob_{pos}'], errors='coerce')

    # Fix the is_correct column
    df['is_correct_fixed'] = (df['predicted_answer'] == df['correct_position']).astype(int)

    return df


def analyze_positional_bias_statistics(df: pd.DataFrame,
                                       dataset_name: str,
                                       model_name: str,
                                       stat_dir: Path,
                                       source_csv: str,
                                       prediction_column: str = "predicted_answer",
                                       correct_column: str = "correct_position",
                                       analysis_label: str = None,
                                       output_filename_base: str = None,
                                       analysis_file_path: Path = None,
                                       write_mode: str = "w",
                                       print_output: bool = True) -> Tuple[Path, List[str]]:
    """Generate positional bias statistics and persist them to disk."""
    if analysis_file_path is None:
        stat_dir.mkdir(parents=True, exist_ok=True)
        if not output_filename_base:
            output_filename_base = f"{dataset_name}-{model_name}_baseline"
        analysis_file_path = stat_dir / f"{output_filename_base}_analysis.txt"
    else:
        analysis_file_path = Path(analysis_file_path)
        analysis_file_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines: List[str] = []

    def _log(message: str):
        if print_output:
            print(message)
        report_lines.append(str(message))

    if analysis_label:
        header = (f"\n=== POSITIONAL BIAS ANALYSIS for {model_name} "
                  f"(dataset={dataset_name}) [{analysis_label}] ===")
    else:
        header = f"\n=== POSITIONAL BIAS ANALYSIS for {model_name} (dataset={dataset_name}) ==="

    _log(header)
    _log(f"Source CSV: {source_csv}")

    df_local = df.copy()

    if prediction_column not in df_local.columns or correct_column not in df_local.columns:
        _log(f"ERROR: Required columns '{prediction_column}' or '{correct_column}' missing.")
        file_exists = analysis_file_path.exists()
        with open(analysis_file_path, write_mode, encoding="utf-8") as f:
            if write_mode == "a" and file_exists and f.tell() > 0:
                f.write("\n")
            f.write("\n".join(report_lines))
        print(f"\nPositional bias analysis saved to: {analysis_file_path}")
        return analysis_file_path, report_lines

    df_local[prediction_column] = df_local[prediction_column].astype(str).str.upper().str.strip()
    df_local[correct_column] = df_local[correct_column].astype(str).str.upper().str.strip()

    valid_responses = df_local[df_local[prediction_column].isin(["A", "B", "C", "D"])].copy()
    failed_responses = len(df_local) - len(valid_responses)

    if failed_responses > 0:
        _log(f"WARNING: {failed_responses}/{len(df_local)} responses failed to parse")

    if len(valid_responses) == 0:
        _log("ERROR: No valid responses to analyze")
        file_exists = analysis_file_path.exists()
        with open(analysis_file_path, write_mode, encoding="utf-8") as f:
            if write_mode == "a" and file_exists and f.tell() > 0:
                f.write("\n")
            f.write("\n".join(report_lines))
        print(f"\nPositional bias analysis saved to: {analysis_file_path}")
        return analysis_file_path, report_lines

    valid_responses["is_correct_eval"] = (
        valid_responses[prediction_column] == valid_responses[correct_column]
    ).astype(int)

    choice_counts = valid_responses[prediction_column].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    total_valid = len(valid_responses)

    _log(f"\nCHOICE DISTRIBUTION (n={total_valid}):")
    for letter in ["A", "B", "C", "D"]:
        count = choice_counts[letter]
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        _log(f"   {letter}: {count:4d} ({percentage:5.1f}%)")

    expected_per_choice = total_valid / 4 if total_valid > 0 else 0
    expected = [expected_per_choice] * 4
    chi2_stat, p_value = chisquare(choice_counts.values, f_exp=expected) if total_valid > 0 else (0.0, 1.0)

    _log(f"\nCHI-SQUARE TEST vs Uniform Distribution:")
    _log(f"   Chi-square statistic: {chi2_stat:.3f}")
    _log(f"   P-value: {p_value:.6f}")
    if p_value < 0.05:
        _log("   Significant deviation from uniform (p < 0.05) - OVERALL BIAS DETECTED")
    else:
        _log("   No significant overall deviation from uniform (p >= 0.05)")

    _log("\nPAIRWISE CHOICE COMPARISONS (Bonferroni-corrected):")
    letters = ["A", "B", "C", "D"]
    comparisons = list(combinations(letters, 2))
    alpha = 0.05
    corrected_alpha = alpha / len(comparisons) if comparisons else alpha
    significant_pairs = 0

    for pos1, pos2 in comparisons:
        count1 = choice_counts[pos1]
        count2 = choice_counts[pos2]
        if count1 + count2 > 0:
            _, p_pair = chisquare([count1, count2])
            if p_pair < corrected_alpha:
                significant_pairs += 1
                _log(f"   - {pos1} vs {pos2}: Significant difference (p={p_pair:.4f} < {corrected_alpha:.4f})")

    if significant_pairs == 0:
        _log("   No pairs showed significant differences in choice frequency.")

    _log(f"\nACCURACY BY CORRECT ANSWER POSITION:")
    accuracy_by_position = valid_responses.groupby(correct_column)["is_correct_eval"].agg(['mean', 'count'])

    overall_accuracy = valid_responses["is_correct_eval"].mean()
    _log(f"   Overall accuracy: {overall_accuracy:.3f}")
    _log(f"   Position-specific accuracy:")

    for letter in letters:
        if letter in accuracy_by_position.index:
            acc = accuracy_by_position.loc[letter, "mean"]
            count = accuracy_by_position.loc[letter, "count"]
            diff = acc - overall_accuracy
            _log(f"     {letter}: {acc:.3f} (n={count}, diff={diff:+.3f})")
        else:
            _log(f"     {letter}: N/A (no questions)")

    _log("\nCHI-SQUARE TEST for Accuracy vs Position:")
    try:
        contingency_table = pd.crosstab(valid_responses[correct_column], valid_responses["is_correct_eval"])
        contingency_table = contingency_table.reindex(letters, fill_value=0)
        if 0 not in contingency_table.columns:
            contingency_table[0] = 0
        if 1 not in contingency_table.columns:
            contingency_table[1] = 0
        contingency_table = contingency_table.rename(columns={0: "Incorrect", 1: "Correct"})[["Correct", "Incorrect"]]

        if contingency_table.sum().sum() > 0 and not (contingency_table.sum(axis=0) == 0).any() and not (contingency_table.sum(axis=1) == 0).any():
            chi2_acc, p_acc, _, _ = chi2_contingency(contingency_table)
            _log(f"   Contingency Table:\n{contingency_table.to_string(header=True)}")
            _log(f"   Chi-square statistic: {chi2_acc:.3f}")
            _log(f"   P-value: {p_acc:.6f}")
            if p_acc < 0.05:
                _log("   Significant relationship between accuracy and position (p < 0.05) - ACCURACY BIAS DETECTED")
            else:
                _log("   No significant relationship between accuracy and position (p >= 0.05)")
        else:
            _log("   Skipped: Not enough data diversity for a valid test.")
            _log(f"   Contingency Table:\n{contingency_table.to_string(header=True)}")
    except Exception as e:
        _log(f"   Could not perform chi-square test for accuracy: {e}")

    choice_percentages = choice_counts.values / total_valid * 100 if total_valid > 0 else np.zeros(4)
    position_bias_score = np.std(choice_percentages)
    _log(f"\nPOSITION BIAS SCORE: {position_bias_score:.2f}")
    _log("   (Standard deviation of choice percentages - higher = more biased)")

    file_exists = analysis_file_path.exists()
    with open(analysis_file_path, write_mode, encoding="utf-8") as f:
        if write_mode == "a" and file_exists and f.tell() > 0:
            f.write("\n")
        f.write("\n".join(report_lines))

    if print_output:
        print(f"\nPositional bias analysis saved to: {analysis_file_path}")
    return analysis_file_path, report_lines


def analyze_bias_effect(test_df_original: pd.DataFrame,
                        test_df_debiased: pd.DataFrame) -> Dict:
    """Analyze where debiasing helped vs. hurt."""

    test_df_original = test_df_original.copy()
    test_df_original['is_correct_fixed'] = (
        test_df_original['predicted_answer'] == test_df_original['correct_position']
    ).astype(int)

    analysis = {}

    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: WHERE DID DEBIASING HELP/HURT?")
    print("=" * 70)

    # 1. Accuracy by correct answer position
    print("\n📍 Accuracy by Correct Answer Position:")
    print(f"{'Position':<12} {'Original':>12} {'Debiased':>12} {'Change':>12} {'N':>8}")
    print("-" * 70)

    for pos in ['A', 'B', 'C', 'D']:
        pos_mask = test_df_original['correct_position'] == pos

        orig_acc = test_df_original[pos_mask]['is_correct_fixed'].mean()
        deb_acc = test_df_debiased[pos_mask]['debiased_is_correct'].mean()

        print(f"{pos:<12} {orig_acc:>11.4f} {deb_acc:>11.4f} "
              f"{deb_acc - orig_acc:>+11.4f} {pos_mask.sum():>8}")

        analysis[f'acc_when_correct_is_{pos}'] = {
            'original': orig_acc,
            'debiased': deb_acc,
            'change': deb_acc - orig_acc,
            'n': pos_mask.sum()
        }

    # 2. Changes breakdown
    print("\n📊 Changes Breakdown:")

    improved = (
        (test_df_debiased['debiased_is_correct'] == 1) &
        (test_df_original['is_correct_fixed'] == 0)
    )

    worsened = (
        (test_df_debiased['debiased_is_correct'] == 0) &
        (test_df_original['is_correct_fixed'] == 1)
    )

    unchanged_correct = (
        (test_df_debiased['debiased_is_correct'] == 1) &
        (test_df_original['is_correct_fixed'] == 1)
    )

    unchanged_wrong = (
        (test_df_debiased['debiased_is_correct'] == 0) &
        (test_df_original['is_correct_fixed'] == 0)
    )

    print(f"  ✅ Improved (wrong → correct): {improved.sum():>4} samples")
    print(f"  ❌ Worsened (correct → wrong): {worsened.sum():>4} samples")
    print(f"  ✓  Stayed correct:             {unchanged_correct.sum():>4} samples")
    print(f"  ✗  Stayed wrong:               {unchanged_wrong.sum():>4} samples")
    print(f"  📈 Net change: {improved.sum() - worsened.sum():+d} samples")

    analysis['changes'] = {
        'improved': int(improved.sum()),
        'worsened': int(worsened.sum()),
        'unchanged_correct': int(unchanged_correct.sum()),
        'unchanged_wrong': int(unchanged_wrong.sum()),
        'net': int(improved.sum() - worsened.sum())
    }

    # 3. Where did improvements come from?
    print("\n🎯 Where Improvements Came From:")
    print(f"{'Correct Pos':<15} {'Improved':>12} {'Total at Pos':>15} {'Rate':>12}")
    print("-" * 70)

    for pos in ['A', 'B', 'C', 'D']:
        pos_mask = test_df_original['correct_position'] == pos
        improved_at_pos = (improved & pos_mask).sum()
        total_at_pos = pos_mask.sum()
        rate = improved_at_pos / total_at_pos if total_at_pos > 0 else 0

        print(f"{pos:<15} {improved_at_pos:>12} {total_at_pos:>15} {rate:>11.2%}")

    # 4. Where did losses come from?
    print("\n💔 Where Losses Came From:")
    print(f"{'Correct Pos':<15} {'Worsened':>12} {'Total at Pos':>15} {'Rate':>12}")
    print("-" * 70)

    for pos in ['A', 'B', 'C', 'D']:
        pos_mask = test_df_original['correct_position'] == pos
        worsened_at_pos = (worsened & pos_mask).sum()
        total_at_pos = pos_mask.sum()
        rate = worsened_at_pos / total_at_pos if total_at_pos > 0 else 0

        print(f"{pos:<15} {worsened_at_pos:>12} {total_at_pos:>15} {rate:>11.2%}")

    return analysis


def check_answer_key_distribution(df: pd.DataFrame):
    """Check if the answer key has uniform distribution."""

    print("\n" + "=" * 70)
    print("ANSWER KEY DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Original correct answers (before permutation)
    original_dist = df.groupby('question_id')['original_correct'].first().value_counts()

    print("\n📝 Original Answer Key Distribution:")
    print(f"{'Answer':<10} {'Count':>10} {'Percentage':>12}")
    print("-" * 70)

    total = original_dist.sum()
    for answer in ['A', 'B', 'C', 'D']:
        count = original_dist.get(answer, 0)
        pct = count / total
        print(f"{answer:<10} {count:>10} {pct:>11.2%}")

    # Statistical test for uniformity
    expected = total / 4
    chi_square = sum((original_dist.get(ans, 0) - expected)**2 / expected
                     for ans in ['A', 'B', 'C', 'D'])

    print(f"\n📊 Chi-square test for uniformity: {chi_square:.4f}")
    print(f"   Critical value (α=0.05, df=3): 7.815")

    if chi_square > 7.815:
        print("   ❌ Answer key is NOT uniformly distributed")
        print("   → This explains why positional bias can be helpful")
    else:
        print("   ✅ Answer key is approximately uniform")

    # Check per permutation
    print("\n📍 Correct Answer Distribution by Permutation:")
    print(f"{'Perm':<6} {'A':>8} {'B':>8} {'C':>8} {'D':>8}")
    print("-" * 40)

    for perm_idx in range(4):
        perm_data = df[df['permutation_idx'] == perm_idx]
        dist = perm_data['correct_position'].value_counts()

        print(f"{perm_idx:<6}", end="")
        for pos in ['A', 'B', 'C', 'D']:
            print(f"{dist.get(pos, 0):>8}", end="")
        print()


def test_alpha_values(df: pd.DataFrame, calibration_ratio: float = 0.20):
    """Test different alpha values to find optimal debiasing strength."""

    print("\n" + "=" * 70)
    print(f"TESTING ALPHA VALUES (Calibration={calibration_ratio:.0%})")
    print("=" * 70)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print("\nTesting alphas:", alphas)
    print("(This may take a minute...)\n")

    for alpha in alphas:
        pride = PriDeDebiasing(
            calibration_ratio=calibration_ratio,
            alpha=alpha,
            random_seed=42
        )

        test_df_debiased, calibration_info = pride.fit_and_predict(df)
        _, test_df_original = pride.split_calibration_test(df)

        # Calculate accuracy
        test_df_original['is_correct_fixed'] = (
            test_df_original['predicted_answer'] == test_df_original['correct_position']
        ).astype(int)

        orig_acc = test_df_original['is_correct_fixed'].mean()
        deb_acc = test_df_debiased['debiased_is_correct'].mean()

        # Position distribution variance
        deb_dist = test_df_debiased['debiased_predicted_answer'].value_counts(normalize=True)
        pos_variance = deb_dist.var()

        results.append({
            'alpha': alpha,
            'original_acc': orig_acc,
            'debiased_acc': deb_acc,
            'improvement': deb_acc - orig_acc,
            'pos_variance': pos_variance
        })

    # Print results
    print(f"\n{'Alpha':<8} {'Original':>12} {'Debiased':>12} {'Improvement':>12} {'Pos Var':>12}")
    print("-" * 70)

    for r in results:
        marker = " 🏆" if r['debiased_acc'] == max(res['debiased_acc'] for res in results) else ""
        print(f"{r['alpha']:<8.2f} {r['original_acc']:>11.4f} {r['debiased_acc']:>11.4f} "
              f"{r['improvement']:>+11.4f} {r['pos_variance']:>11.6f}{marker}")

    # Find best alpha
    best = max(results, key=lambda x: x['debiased_acc'])
    print(f"\n✅ Best alpha: {best['alpha']:.2f} "
          f"(Accuracy: {best['debiased_acc']:.4f}, Δ: {best['improvement']:+.4f})")

    # Analyze best alpha in detail
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS FOR ALPHA={best['alpha']:.2f}")
    print("=" * 70)

    pride_best = PriDeDebiasing(
        calibration_ratio=calibration_ratio,
        alpha=best['alpha'],
        random_seed=42
    )

    test_df_debiased_best, _ = pride_best.fit_and_predict(df)
    _, test_df_original_best = pride_best.split_calibration_test(df)

    analysis = analyze_bias_effect(test_df_original_best, test_df_debiased_best)

    return results, best['alpha']


def check_prior_stability_with_alpha(df: pd.DataFrame,
                                     calibration_ratio: float = 0.20,
                                     alpha: float = 0.5,
                                     n_runs: int = 10):
    """Check stability with specific alpha value."""

    print("\n" + "=" * 70)
    print(f"PRIOR STABILITY ANALYSIS (Alpha={alpha:.2f}, {n_runs} runs)")
    print("=" * 70)

    priors = []
    accuracies = []

    print("\nRunning stability tests...")

    for seed in range(n_runs):
        pride = PriDeDebiasing(
            calibration_ratio=calibration_ratio,
            alpha=alpha,
            random_seed=seed
        )

        test_df_debiased, calibration_info = pride.fit_and_predict(df)

        priors.append(calibration_info['estimated_prior'])
        accuracy = test_df_debiased['debiased_is_correct'].mean()
        accuracies.append(accuracy)

    priors = np.array(priors)
    accuracies = np.array(accuracies)

    print(f"\n📊 Prior Estimates:")
    print(f"{'Position':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV':>10}")
    print("-" * 70)

    for i, pos in enumerate(['A', 'B', 'C', 'D']):
        mean = priors[:, i].mean()
        std = priors[:, i].std()
        cv = std / mean if mean > 0 else 0

        print(f"{pos:<10} {mean:>10.4f} {std:>10.4f} {priors[:, i].min():>10.4f} "
              f"{priors[:, i].max():>10.4f} {cv:>9.2%}")

    print(f"\n📈 Resulting Accuracies:")
    print(f"   Mean:  {accuracies.mean():.4f}")
    print(f"   Std:   {accuracies.std():.4f}")
    print(f"   Range: [{accuracies.min():.4f}, {accuracies.max():.4f}]")
    print(f"   CV:    {accuracies.std()/accuracies.mean():.2%}")


def main_comprehensive(csv_path: str):
    """Comprehensive analysis with all diagnostics."""

    print("=" * 70)
    print("PriDe: COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Date: 2025-10-31 16:57:58 UTC")
    print(f"User: h4yd3nt4ng")
    print(f"Dataset: {csv_path}")
    print("=" * 70)

    dataset_name = Path(csv_path).stem

    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data(csv_path)

    print(f"✅ Loaded successfully!")
    print(f"   Total questions: {df['question_id'].nunique()}")
    print(f"   Total samples: {len(df)}")
    print(f"   Permutations per question: {df.groupby('question_id').size().mode()[0]}")

    if 'model' in df.columns and df['model'].notna().any():
        model_name_raw = str(df['model'].dropna().iloc[0])
    else:
        model_name_raw = "unknown_model"
    model_name = model_name_raw.replace(':', '_').replace('/', '_')

    stat_dir = Path("results") / "pride_stat_results"
    analyze_positional_bias_statistics(
        df=df,
        dataset_name=dataset_name,
        model_name=model_name,
        stat_dir=stat_dir,
        source_csv=csv_path
    )

    # 1. Check answer key distribution
    check_answer_key_distribution(df)

    # 2. Test different alpha values
    results, best_alpha = test_alpha_values(df, calibration_ratio=0.20)

    # 3. Stability check for best alpha
    check_prior_stability_with_alpha(df, calibration_ratio=0.20,
                                     alpha=best_alpha, n_runs=10)

    # 4. Final recommendation
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    print(f"\n✅ Optimal Configuration:")
    print(f"   - Calibration Ratio: 20%")
    print(f"   - Alpha (debiasing strength): {best_alpha:.2f}")

    best_result = [r for r in results if r['alpha'] == best_alpha][0]
    print(f"   - Expected accuracy: {best_result['debiased_acc']:.4f}")
    print(f"   - Accuracy improvement: {best_result['improvement']:+.4f}")
    print(f"   - Position variance: {best_result['pos_variance']:.6f}")

    print(f"\n📊 Interpretation:")
    if best_alpha < 0.3:
        print("   ⚠️  Very low alpha - The model's positional bias is mostly helpful")
        print("       Only minimal debiasing is recommended")
        print("       This suggests the answer key may favor certain positions")
    elif best_alpha < 0.7:
        print("   ⚖️  Moderate alpha - The model's bias is mixed (helpful + harmful)")
        print("       Partial debiasing provides best balance")
        print("       PriDe successfully optimizes the trade-off")
    else:
        print("   ✅ High alpha - The model's bias is mostly harmful")
        print("       Strong debiasing is beneficial")
        print("       The answer key is likely uniform or the bias is severe")

    print(f"\n💡 Usage Instructions:")
    print(f"   To apply PriDe with optimal settings:")
    print(f"   ```python")
    print(f"   pride = PriDeDebiasing(calibration_ratio=0.20, alpha={best_alpha:.2f}, random_seed=42)")
    print(f"   test_df_debiased, info = pride.fit_and_predict(df)")
    print(f"   ```")

    # 5. Save best model results and summary report
    print(f"\n📁 Saving results...")

    # Rerun with the best alpha to get the final debiased dataframe and info
    pride_final = PriDeDebiasing(
        calibration_ratio=0.20,
        alpha=best_alpha,
        random_seed=42
    )
    test_df_debiased_final, calibration_info_final = pride_final.fit_and_predict(df)

    # Define output paths
    input_path = Path(csv_path)
    csv_output_dir = Path("results") / "pride_optimized_csv_results"
    stat_output_dir = Path("results") / "pride_optimized_stat_results"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    stat_output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = input_path.stem
    
    # Path for the debiased data CSV
    output_csv_path = csv_output_dir / f"{output_stem}_pride_debiased.csv"
    test_df_debiased_final.to_csv(output_csv_path, index=False)

    print(f"✅ Debiased results saved to:")
    print(f"   {output_csv_path}")

    # Path for the summary report TXT
    report_path = stat_output_dir / f"{output_stem}_pride_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PriDe Debiasing Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Source Dataset: {csv_path}\n\n")
        f.write(f"Optimal Configuration:\n")
        f.write(f"  Calibration Ratio: 20%\n")
        f.write(f"  Alpha (Debiasing Strength): {best_alpha:.2f}\n")
        f.write(f"  Calibration Questions: {calibration_info_final['n_calibration_questions']}\n")
        f.write(f"  Test Questions: {calibration_info_final['n_test_questions']}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Original Accuracy (from argmax): {best_result['original_acc']:.4f}\n")
        f.write(f"  Debiased Accuracy: {best_result['debiased_acc']:.4f}\n")
        f.write(f"  Improvement: {best_result['improvement']:+.4f}\n")
        f.write(f"  Position Variance (after debias): {best_result['pos_variance']:.6f}\n\n")
        f.write(f"Estimated Prior (Model Bias):\n")
        for i, pos in enumerate(['A', 'B', 'C', 'D']):
            f.write(f"  Position {pos}: {calibration_info_final['estimated_prior'][i]:.4f}\n")

    print(f"✅ Summary report saved to:")
    print(f"   {report_path}")

    debiased_df_for_stats = pd.read_csv(output_csv_path)
    debiased_filename_base = f"{output_stem}-{model_name}_pride_debiased_alpha{best_alpha:.2f}"
    debiased_stats_path, debiased_report_lines = analyze_positional_bias_statistics(
        df=debiased_df_for_stats,
        dataset_name=dataset_name,
        model_name=model_name,
        stat_dir=stat_output_dir,
        source_csv=str(output_csv_path),
        prediction_column="debiased_predicted_answer",
        analysis_label=f"PriDe Debiased (alpha={best_alpha:.2f})",
        output_filename_base=debiased_filename_base
    )

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n")
        f.write("\n".join(debiased_report_lines))
        f.write("\n")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE! 🎉")
    print("=" * 70)

    return results, best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PriDe debiasing and comprehensive analysis.")
    parser.add_argument("csv_path", help="Path to the input CSV with model probabilities.")
    args = parser.parse_args()
    results, best_alpha = main_comprehensive(args.csv_path)
