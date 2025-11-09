# LLM Positional Bias Evaluation and Debiasing

## Overview

This project investigates positional bias in Large Language Models (LLMs) when answering multiple-choice questions. It provides tools to:
1.  Evaluate the extent of positional bias in various models using sampling-based methods.
2.  Apply and analyze the effectiveness of the PriDe (Prior Debiasing) method.
3.  Visualize the results to compare model performance before and after debiasing.

## Project Workflow

The project follows a three-step workflow:

1.  **Generate Raw Data**: Use `eval_sampling_bias.py` to run a model against a dataset of MCQs. This script creates a CSV file containing the model's answers and empirical probabilities derived from sampling.
2.  **Analyze and Debias**: Use `pride_eval.py` to analyze the generated CSV. This script applies the PriDe algorithm to the data, calculates the improvement, and saves a debiased CSV and a summary report.
3.  **Visualize Results**: Use `plot_results.py` to generate a bar chart comparing the original and debiased accuracies for all tested models.

## Key Files

-   `eval_sampling_bias.py`: The primary script for evaluating models. It uses parallel sampling to generate robust empirical probabilities.
-   `pride_eval.py`: Analyzes the output from the evaluation script, applies the PriDe debiasing method, and saves detailed reports.
-   `plot_results.py`: Scans the results directory and generates a `pride_effectiveness_comparison.png` bar chart.
-   `eval_positional_bias_copy.py`: An alternative evaluation script that uses the model's self-reported probabilities instead of sampling.
-   `ict_pp/csv/2012-2020_ICT_DSE.csv`: The primary dataset used for evaluation.
-   `results/`: Directory containing all output files.
    -   `temp/`: Contains the latest experimental results, including raw data, analysis reports, and debiased data.

## Usage

### Step 1: Generate Evaluation Data

Run the sampling-based evaluation for a specific model.

```bash
# Example for llama3.2 with 25 samples per permutation
python eval_sampling_bias.py --model llama3.2 --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
```

### Step 2: Run PriDe Debiasing Analysis

Modify `pride_eval.py` to point `csv_path` to the output file from Step 1, then run it.

```python
# Inside pride_eval.py
csv_path = 'results/temp/2012-2020_ICT_DSE-llama3.2_latest_sampling_n25.csv'
```

```bash
python pride_eval.py
```

### Step 3: Visualize All Results

After running the analysis for all models, generate the summary plot.

```bash
python plot_results.py
```
This will create `pride_effectiveness_comparison.png` in the root directory.

## Requirements

See `requirements.txt` for Python dependencies.

## Link to Overleaf

https://www.overleaf.com/project/68be8d02fb134c31cf86398d
