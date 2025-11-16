"""
Local LLM Positional Bias Data Collection (Sampling-Based)
Collects raw evaluation data by empirically estimating probabilities via multiple samples,
as described for models like GPT-3.5-turbo that do not return token probabilities.

This script focuses on data collection only. Run batch_stat_analysis.py afterwards 
to perform comprehensive statistical analysis on the collected CSV data.
"""

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import requests


# Prompt template is simplified as we no longer ask for probabilities
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
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
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
    
    shift = perm_idx % 4
    cyclic_items = original_items[shift:] + original_items[:shift]
    
    new_options = {letters[i]: cyclic_items[i][1] for i in range(4)}
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
                temperature: float = 0.7, seed: int = 42, retries: int = 3, 
                timeout: int = 60) -> str:
    """Call Ollama API to get model response"""
    url = f"{host.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
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


def _sample_worker(model: str, prompt: str, host: str, temperature: float, seed: int, timeout: int) -> str:
    """Helper function for a single sampling call, designed for parallel execution."""
    try:
        response_text = call_ollama(
            model=model, prompt=prompt, host=host,
            temperature=temperature, seed=seed, timeout=timeout
        )
        return parse_answer(response_text)
    except Exception:
        return ""  # Return empty string on failure


def sample_and_estimate_distribution(model: str, prompt: str, host: str, n_samples: int, 
                                     temperature: float, seed: int, timeout: int,
                                     num_workers: int
                                     ) -> Tuple[Dict[str, float], str, Dict[str, int], int]:
    """
    Samples multiple responses in parallel to estimate a probability distribution.
    """
    choices = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a future for each parallel sampling call
        futures = [
            executor.submit(
                _sample_worker, model, prompt, host, temperature, seed + i, timeout
            )
            for i in range(n_samples)
        ]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            answer = future.result()
            if answer in ["A", "B", "C", "D"]:
                choices.append(answer)

    total = len(choices)
    if total == 0:
        # Fallback case if all samples fail
        probabilities = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        predicted_answer = "" 
        raw_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        return probabilities, predicted_answer, raw_counts, 0
        
    counter = Counter(choices)
    raw_counts = {letter: counter.get(letter, 0) for letter in ["A", "B", "C", "D"]}
    probabilities = {letter: count / total for letter, count in raw_counts.items()}
    
    # Predicted answer is the most common sample (the mode)
    predicted_answer = counter.most_common(1)[0][0]
    
    return probabilities, predicted_answer, raw_counts, total


def run_evaluation(model: str, host: str, csv_path: str, n_permutations: int,
                  seed: int, max_questions: int,
                  sampling_n: int, sampling_temp: float, num_workers: int):
    """Run data collection for positional bias evaluation using sampling.
    
    Collects raw evaluation data and saves to CSV. Statistical analysis 
    should be performed separately using batch_stat_analysis.py.
    """
    
    print("\n=== Starting Positional Bias Evaluation (Sampling-Based) ===")
    print(f"Model: {model}")
    print(f"Dataset: {csv_path}")
    print(f"Permutations per question: {n_permutations}")
    print(f"Samples per permutation: {sampling_n}")
    print(f"Sampling Temperature: {sampling_temp}")
    print(f"Parallel workers: {num_workers}")
    
    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)
    
    # --- Checkpointing and Output File Setup ---
    csv_dir = Path("results") / "csv_results"
    csv_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(csv_path).stem
    model_name = model.replace(':', '_').replace('/', '_')
    output_filename_base = f"{dataset_name}-{model_name}_sampling_n{sampling_n}"
    csv_output_file = csv_dir / f"{output_filename_base}.csv"

    processed_tasks = set()
    fieldnames = [
        "question_id", "permutation_idx", "model", "predicted_answer",
        "correct_position", "original_correct", "is_correct",
        "prob_A", "prob_B", "prob_C", "prob_D",
        "sample_count_A", "sample_count_B", "sample_count_C", "sample_count_D",
        "total_valid_samples", "question"
    ]

    if csv_output_file.exists():
        print(f"Resuming from existing file: {csv_output_file}")
        try:
            with open(csv_output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('question_id') and row.get('permutation_idx'):
                        processed_tasks.add((row['question_id'], int(row['permutation_idx'])))
            print(f"Found {len(processed_tasks)} completed tasks to skip.")
        except (csv.Error, KeyError, ValueError) as e:
            print(f"Warning: Could not parse existing results file, starting fresh. Error: {e}")
            processed_tasks = set()


    total_prompts = len(mcqs) * n_permutations
    
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not processed_tasks and f.tell() == 0:
            writer.writeheader()

        with tqdm(total=total_prompts, desc=f"Evaluating {model}", initial=len(processed_tasks)) as pbar:
            for mcq in mcqs:
                for perm_idx in range(n_permutations):
                    if (mcq.uid, perm_idx) in processed_tasks:
                        continue

                    permuted_options, new_to_old_mapping = permute_options(mcq.options, perm_idx)
                    
                    correct_new_position = next(
                        (new_pos for new_pos, old_pos in new_to_old_mapping.items() if old_pos == mcq.answer), None
                    )
                    
                    prompt = build_prompt(mcq, permuted_options)
                    
                    try:
                        probabilities, predicted_answer, raw_counts, total_sampled = sample_and_estimate_distribution(
                            model=model,
                            prompt=prompt,
                            host=host,
                            n_samples=sampling_n,
                            temperature=sampling_temp,
                            seed=seed + perm_idx,
                            timeout=180,
                            num_workers=num_workers
                        )
                        is_correct = (predicted_answer == correct_new_position)
                        
                    except Exception as e:
                        print(f"Error processing {mcq.uid} perm {perm_idx}: {e}")
                        predicted_answer = ""
                        is_correct = False
                        probabilities = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
                        raw_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
                        total_sampled = 0
                    
                    result_row = {
                        "question_id": mcq.uid,
                        "permutation_idx": perm_idx,
                        "model": model,
                        "predicted_answer": predicted_answer,
                        "correct_position": correct_new_position,
                        "original_correct": mcq.answer,
                        "is_correct": int(is_correct),
                        "prob_A": probabilities["A"],
                        "prob_B": probabilities["B"],
                        "prob_C": probabilities["C"],
                        "prob_D": probabilities["D"],
                        "sample_count_A": raw_counts["A"],
                        "sample_count_B": raw_counts["B"],
                        "sample_count_C": raw_counts["C"],
                        "sample_count_D": raw_counts["D"],
                        "total_valid_samples": total_sampled,
                        "question": mcq.question,
                    }
                    writer.writerow(result_row)
                    f.flush() # Write to disk immediately for checkpointing
                    pbar.update(1)

    print(f"\nEvaluation complete. Results saved to: {csv_output_file}")
    print(f"Run batch_stat_analysis.py to perform statistical analysis on the collected data.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate positional bias using sampling-based probability estimation.")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--input", type=str, required=True, help="Path to MCQ CSV file")
    parser.add_argument("--n-permutations", type=int, default=4, help="Number of permutations per question")
    parser.add_argument("--sampling-n", type=int, default=15, help="Number of samples for probability estimation")
    parser.add_argument("--sampling-temp", type=float, default=0.7, help="Sampling temperature (must be > 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to evaluate")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for sampling")
    
    args = parser.parse_args()
    
    if args.sampling_temp <= 0:
        raise ValueError("Sampling temperature must be greater than 0 for this script to work.")
        
    run_evaluation(
        model=args.model,
        host=args.host,
        csv_path=args.input,
        n_permutations=args.n_permutations,
        seed=args.seed,
        max_questions=args.max_questions,
        sampling_n=args.sampling_n,
        sampling_temp=args.sampling_temp,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
