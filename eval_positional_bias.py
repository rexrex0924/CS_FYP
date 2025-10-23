"""
Local LLM Positional Bias Evaluation
Evaluates positional bias in multiple choice questions using local LLM models via Ollama
"""

import argparse
import csv
import json
import re
import time
import os
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency
from tqdm import tqdm
import requests


# Prompt template for multiple choice questions
PROMPT_TEMPLATE = """Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

You must respond with exactly one letter (A, B, C, or D) and nothing else.
Answer:"""

# Regex to extract answer letter
LETTER_RE = re.compile(r'\b([A-D])\b')

@dataclass
class MCQ:
    uid: str
    question: str
    options: Dict[str, str]  # keys A-D, values are option text
    answer: str  # correct letter A-D


def load_mcq_csv(path: str, max_questions: int = None) -> List[MCQ]:
    """Load multiple choice questions from CSV file"""
    df = pd.read_csv(path)
    required_cols = {"id", "question", "option_a", "option_b", "option_c", "option_d", "answer"}
                
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in CSV: {missing}")

    mcqs = []
    for _, row in df.iterrows():
        ans = str(row["answer"]).strip().upper()
        if ans not in {"A", "B", "C", "D"}:
            print(f"Skipping question {row['id']} - invalid answer: {ans}")
            continue
        
        # Check for missing question or options
        question = str(row["question"]).strip()
        options_text = [
            str(row["option_a"]).strip(),
            str(row["option_b"]).strip(), 
            str(row["option_c"]).strip(),
            str(row["option_d"]).strip()
        ]
        
        if (question in ["", "nan"] or 
            any(opt in ["", "nan"] for opt in options_text)):
            print(f"Skipping question {row['id']} - incomplete data")
            continue
            
        mcq = MCQ(
            uid=str(row["id"]),
            question=question,
            options={
                "A": options_text[0],
                "B": options_text[1],
                "C": options_text[2],
                "D": options_text[3],
            },
            answer=ans,
        )
        mcqs.append(mcq)
    
    if max_questions:
        mcqs = mcqs[:max_questions]
    
    print(f"Loaded {len(mcqs)} questions from {path}")
    return mcqs


def permute_options(options: Dict[str, str], perm_idx: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Cyclically permute the options and return the new mapping
    Returns: (new_options, mapping from new_letter -> original_letter)
    """
    letters = ["A", "B", "C", "D"]
    original_items = [(letter, options[letter]) for letter in letters]
    
    # Cycle through permutations: shift by (perm_idx % 4)
    shift = perm_idx % 4
    cyclic_items = original_items[shift:] + original_items[:shift]
    
    # Create new mapping
    new_options = {letters[i]: cyclic_items[i][1] for i in range(4)}
    # Track which new position corresponds to which original position
    new_to_old_mapping = {letters[i]: cyclic_items[i][0] for i in range(4)}
    
    return new_options, new_to_old_mapping


def build_prompt(mcq: MCQ, permuted_options: Dict[str, str]) -> str:
    """Build the full prompt for the LLM"""
    return PROMPT_TEMPLATE.format(
        question=mcq.question,
        A=permuted_options["A"],
        B=permuted_options["B"],
        C=permuted_options["C"],
        D=permuted_options["D"],
    )


def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434", 
                temperature: float = 0.0, seed: int = 42, retries: int = 3, 
                timeout: int = 60) -> str:
    """Call Ollama API to get model response"""
    url = f"{host.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0,
            "seed": seed,
            "num_ctx": 4096,  # Increased context length
            "stop": ["Question:", "Options:", "\n\nQuestion:"]  # Removed "\n\n" stop condition
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            text = data.get("response", "").strip()
            
            if not text:
                raise RuntimeError("Empty response from model")
                
            return text
            
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt * 0.5)
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1.0)
    
    return ""


def parse_answer(response_text: str) -> str:
    """Extract the answer letter from model response"""
    response_text = response_text.strip().upper()
    
    # Check if response contains think tags (even incomplete ones)
    if "<THINK>" in response_text:
        # Extract text from opening think tag to end (handle incomplete responses)
        think_start = response_text.find("<THINK>")
        if "</THINK>" in response_text:
            think_end = response_text.find("</THINK>")
            think_content = response_text[think_start:think_end]
        else:
            # Handle incomplete think blocks
            think_content = response_text[think_start:]
        
        # Enhanced answer patterns for qwen model's reasoning style
        answer_patterns = [
            r'ANSWER\s+(?:SHOULD\s+BE|IS|MUST\s+BE)\s+([A-D])',
            r'SO\s+(?:THE\s+)?ANSWER\s+(?:SHOULD\s+BE|IS|MUST\s+BE)\s+([A-D])',
            r'(?:DEFINITELY|CLEARLY)\s+([A-D])',
            r'(?:SO|THEREFORE),?\s+(?:THE\s+ANSWER\s+IS\s+)?([A-D])',
            r'OPTION\s+([A-D])',
            r'CHOICE\s+([A-D])',
            r'([A-D]),?\s+(?:IS\s+THE\s+ANSWER|IS\s+CORRECT)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, think_content)
            if match:
                return match.group(1)
    
    # Enhanced fallback: look for letters in context
    # Prioritize letters that appear after reasoning words
    reasoning_patterns = [
        r'(?:ANSWER|OPTION|CHOICE|SO|THEREFORE)\s+(?:IS\s+)?([A-D])',
        r'([A-D])\s+(?:IS\s+(?:THE\s+)?(?:ANSWER|CORRECT|RIGHT))',
        r'MUST\s+BE\s+([A-D])'
    ]
    
    for pattern in reasoning_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            return matches[-1]  # Return the last match
    
    # Final fallback: look for standalone letters (but prefer later ones)
    matches = re.findall(r'\b([A-D])\b', response_text)
    if matches:
        return matches[-1]  # Return the last mentioned letter
    
    return ""

def run_evaluation(model: str, host: str, csv_path: str, n_permutations: int,
                  temperature: float, seed: int, max_questions: int,
                  output_prefix: str):
    """Run the full positional bias evaluation"""
    
    print(f"\n=== Starting Positional Bias Evaluation ===")
    print(f"Model: {model}")
    print(f"Host: {host}")
    print(f"Dataset: {csv_path}")
    print(f"Permutations per question: {n_permutations}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    
    # Note: We no longer need the random number generator for shuffling
    # but we still use seed for the model's generation consistency
    
    # Load questions
    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)
    
    # Test Ollama connection
    try:
        test_response = requests.get(f"{host}/api/tags", timeout=10)
        test_response.raise_for_status()
        print(f"Successfully connected to Ollama at {host}")
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        print("Make sure Ollama is running and accessible")
        return
    
    # Run evaluation
    results = []
    total_prompts = len(mcqs) * n_permutations
    
    with tqdm(total=total_prompts, desc=f"Evaluating {model}") as pbar:
        for mcq in mcqs:
            for perm_idx in range(n_permutations):
                # Create permuted version of options
                permuted_options, new_to_old_mapping = permute_options(mcq.options, perm_idx)
                
                # Find where the correct answer ended up
                correct_new_position = None
                for new_pos, old_pos in new_to_old_mapping.items():
                    if old_pos == mcq.answer:
                        correct_new_position = new_pos
                        break
                
                # Build prompt and get model response
                prompt = build_prompt(mcq, permuted_options)
                
                try:
                    response_text = call_ollama(
                        model=model,
                        prompt=prompt,
                        host=host,
                        temperature=temperature,
                        seed=seed + perm_idx,
                        timeout=120
                    )
                    
                    predicted_answer = parse_answer(response_text)
                    is_correct = (predicted_answer == correct_new_position)
                    
                except Exception as e:
                    print(f"Error processing {mcq.uid} perm {perm_idx}: {e}")
                    response_text = ""
                    predicted_answer = ""
                    is_correct = False
                
                # Store result
                results.append({
                    "question_id": mcq.uid,
                    "permutation_idx": perm_idx,
                    "model": model,
                    "predicted_answer": predicted_answer,
                    "correct_position": correct_new_position,
                    "original_correct": mcq.answer,
                    "is_correct": int(is_correct),
                    "raw_response": response_text.replace('\n', ' ').replace('\r', ''),
                    "question": mcq.question,
                    "option_A": permuted_options["A"],
                    "option_B": permuted_options["B"],
                    "option_C": permuted_options["C"],
                    "option_D": permuted_options["D"],
                })
                
                pbar.update(1)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    
    csv_dir = os.path.join("results", "csv_results")
    stat_dir = os.path.join("results", "stat_results")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    # Extract dataset name from csv_path (without extension)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Format model name (replace : and / with _)
    model_name = model.replace(':', '_').replace('/', '_')
    
    # Create output filename: dataset-model
    output_filename_base = f"{dataset_name}-{model_name}"
    
    csv_output_file = os.path.join(csv_dir, f"{output_filename_base}.csv")
    df.to_csv(csv_output_file, index=False)
    
    # Analyze results
    analyze_results(df, model, csv_output_file, stat_dir, output_filename_base)


def analyze_results(df: pd.DataFrame, model: str, output_file: str, stat_dir: str, output_filename_base: str):
    """Analyze and print results of positional bias evaluation"""
    
    analysis_file_path = os.path.join(stat_dir, f"{output_filename_base}_analysis.txt")
    report_lines = []

    def _log(message):
        print(message)
        report_lines.append(str(message))

    _log(f"\n=== POSITIONAL BIAS ANALYSIS for {model} ===")
    
    # Filter out failed responses
    valid_responses = df[df["predicted_answer"].isin(["A", "B", "C", "D"])]
    failed_responses = len(df) - len(valid_responses)
    
    if failed_responses > 0:
        _log(f"WARNING: {failed_responses}/{len(df)} responses failed to parse")
    
    if len(valid_responses) == 0:
        _log("ERROR: No valid responses to analyze")
        # Save what we have and exit
        with open(analysis_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\nAnalysis report saved to: {analysis_file_path}")
        return
    
    # Overall choice distribution
    choice_counts = valid_responses["predicted_answer"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    total_valid = len(valid_responses)
    
    _log(f"\nCHOICE DISTRIBUTION (n={total_valid}):")
    for letter in ["A", "B", "C", "D"]:
        count = choice_counts[letter]
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        _log(f"   {letter}: {count:4d} ({percentage:5.1f}%)")
    
    # Chi-square test against uniform distribution
    expected_per_choice = total_valid / 4
    expected = [expected_per_choice] * 4
    chi2_stat, p_value = chisquare(choice_counts.values, f_exp=expected)
    
    _log(f"\nCHI-SQUARE TEST vs Uniform Distribution:")
    _log(f"   Chi-square statistic: {chi2_stat:.3f}")
    _log(f"   P-value: {p_value:.6f}")
    if p_value < 0.05:
        _log("   Significant deviation from uniform (p < 0.05) - OVERALL BIAS DETECTED")
    else:
        _log("   No significant overall deviation from uniform (p >= 0.05)")

    # Pairwise comparisons for choice distribution
    _log("\nPAIRWISE CHOICE COMPARISONS (Bonferroni-corrected):")
    letters = ["A", "B", "C", "D"]
    comparisons = list(combinations(letters, 2))
    alpha = 0.05
    corrected_alpha = alpha / len(comparisons)
    significant_pairs = 0
    
    for pos1, pos2 in comparisons:
        count1 = choice_counts[pos1]
        count2 = choice_counts[pos2]
        
        # Test if the counts between two positions are significantly different
        if count1 + count2 > 0:
            _, p_pair = chisquare([count1, count2])
            if p_pair < corrected_alpha:
                significant_pairs += 1
                _log(f"   - {pos1} vs {pos2}: Significant difference (p={p_pair:.4f} < {corrected_alpha:.4f})")

    if significant_pairs == 0:
        _log("   No pairs showed significant differences in choice frequency.")
    
    # Accuracy by position of correct answer
    _log(f"\nACCURACY BY CORRECT ANSWER POSITION:")
    accuracy_by_position = valid_responses.groupby("correct_position")["is_correct"].agg(['mean', 'count'])
    
    overall_accuracy = valid_responses["is_correct"].mean()
    _log(f"   Overall accuracy: {overall_accuracy:.3f}")
    _log(f"   Position-specific accuracy:")
    
    for letter in ["A", "B", "C", "D"]:
        if letter in accuracy_by_position.index:
            acc = accuracy_by_position.loc[letter, "mean"]
            count = accuracy_by_position.loc[letter, "count"]
            diff = acc - overall_accuracy
            _log(f"     {letter}: {acc:.3f} (n={count}, diff={diff:+.3f})")
        else:
            _log(f"     {letter}: N/A (no questions)")

    # Chi-square test for independence of accuracy and position
    p_acc = 1.0  # Default value if test is skipped
    _log("\nCHI-SQUARE TEST for Accuracy vs Position:")
    try:
        # H0: Accuracy is independent of the correct answer's position
        contingency_table = pd.crosstab(valid_responses['correct_position'], valid_responses['is_correct'])
        contingency_table = contingency_table.reindex(["A", "B", "C", "D"], fill_value=0)
        
        # Ensure both 'correct' and 'incorrect' columns exist for the test
        if 0 not in contingency_table.columns: contingency_table[0] = 0
        if 1 not in contingency_table.columns: contingency_table[1] = 0
        
        contingency_table = contingency_table.rename(columns={0: 'Incorrect', 1: 'Correct'})[['Correct', 'Incorrect']]
        
        # Test is only valid if there's data and no row/column sums are zero
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
            _log("   Skipped: Not enough data diversity for a valid test (e.g., all answers correct or incorrect).")
            _log(f"   Contingency Table:\n{contingency_table.to_string(header=True)}")
            
    except Exception as e:
        _log(f"   Could not perform chi-square test for accuracy: {e}")
    
    # Position bias score (standard deviation of choice percentages)
    choice_percentages = choice_counts.values / total_valid * 100
    position_bias_score = np.std(choice_percentages)
    _log(f"\nPOSITION BIAS SCORE: {position_bias_score:.2f}")
    _log(f"   (Standard deviation of choice percentages - higher = more biased)")
    
    _log(f"\nFull results saved to: {output_file}")
    
    # Summary
    bias_detected = False
    reasons = []
    if p_value < 0.05:
        bias_detected = True
        reasons.append("overall choice distribution is not uniform")
    if p_acc < 0.05:
        bias_detected = True
        reasons.append("accuracy depends on answer position")
    if position_bias_score > 5: # Keep this as a heuristic measure
        if not bias_detected: # Avoid duplicate reporting if p-value already caught it
            reasons.append(f"choice distribution is imbalanced (score: {position_bias_score:.2f})")
        bias_detected = True

    if bias_detected:
        _log(f"\nCONCLUSION: {model} shows evidence of positional bias.")
        _log(f"   Reasons: {'; '.join(reasons)}.")
    else:
        _log(f"\nCONCLUSION: {model} shows minimal positional bias.")

    # Save the full report to a file
    with open(analysis_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nAnalysis report saved to: {analysis_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate positional bias in local LLM models")
    parser.add_argument("--model", type=str, required=True, 
                       help="Ollama model name (e.g., llama3.2, mistral, qwen2.5)")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                       help="Ollama host URL")
    parser.add_argument("--input", type=str, default="ict_pp/csv/2012-2020_ICT_DSE.csv",
                       help="Path to MCQ CSV file")
    parser.add_argument("--n-permutations", type=int, default=4,
                       help="Number of permutations per question")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="Maximum number of questions to evaluate")
    parser.add_argument("--output-prefix", type=str, default="results/positional_bias",
                       help="Output file prefix")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    if args.n_permutations < 1:
        print("Error: n-permutations must be at least 1")
        return
    
    # Run evaluation
    run_evaluation(
        model=args.model,
        host=args.host,
        csv_path=args.input,
        n_permutations=args.n_permutations,
        temperature=args.temperature,
        seed=args.seed,
        max_questions=args.max_questions,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()
