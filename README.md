# LLM Positional Bias Evaluation and Debiasing

## Overview

This project investigates positional bias in Large Language Models (LLMs) when answering multiple-choice questions. It provides a comprehensive pipeline to:
1. Collect raw evaluation data from LLMs using sampling-based methods
2. Perform statistical analysis to quantify positional bias
3. Apply the PriDe (Prior Debiasing) method to reduce bias
4. Generate comprehensive visualizations comparing model performance

## Project Workflow

The project follows a four-step pipeline:

### Step 1: Data Collection (Raw Evaluation)
Use `eval_sampling_bias.py` to evaluate models on MCQ datasets. This script:
- Runs multiple permutations of each question (cycling answer positions)
- Samples multiple responses per permutation to estimate probabilities
- Outputs raw CSV files with predictions and empirical probabilities

### Step 2: Statistical Analysis
Use `batch_stat_analysis.py` to process all collected CSV files. This script:
- Computes choice distribution statistics
- Performs chi-square tests for uniformity and independence
- Calculates position bias scores and recall standard deviations
- Generates analysis reports and pairwise comparison heatmaps

### Step 3: PriDe Debiasing (Batch Processing)
Use `pride_batch_summary.py` to apply PriDe debiasing to all datasets. This script:
- Automatically processes all CSV files in `results/csv_results/`
- Tests multiple alpha values to find optimal debiasing strength
- Generates comparison visualizations grouped by dataset and model
- Outputs debiased results and summary reports

### Step 4: Detailed Visualization (Optional)
Use `pride_detail_eval.py` for in-depth analysis of specific model-dataset pairs. This script:
- Generates comprehensive dashboards with multiple metrics
- Creates alpha selection analysis plots
- Produces publication-ready visualizations

## Key Files

### Evaluation Scripts
- `eval_sampling_bias.py`: Primary data collection script using parallel sampling
- `batch_stat_analysis.py`: Batch statistical analysis of raw evaluation data
- `pride_batch_summary.py`: Batch PriDe debiasing with comprehensive visualizations
- `pride_detail_eval.py`: Detailed visualization for individual model-dataset pairs

### Datasets
- `ict_pp/csv/2012-2020_ICT_DSE.csv`: HKDSE ICT examination questions (2012-2020)
- `mmlu/data/*.csv`: MMLU dataset subsets (college_cs, sociology, etc.)

### Output Directories
- `results/csv_results/`: Raw evaluation CSV files from Step 1
- `results/stat_results/`: Statistical analysis reports and heatmaps from Step 2
- `results/pride_summary/`: PriDe comparison visualizations from Step 3
- `results/pride_visualizations/`: Detailed dashboards from Step 4 (optional)

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Ensure Ollama is running with your desired models
# ollama serve
```

## Usage

### Step 1: Collect Raw Evaluation Data

Run sampling-based evaluation for each model you want to test:

```bash
# Single model evaluation
python eval_sampling_bias.py \
    --model llama3.2:latest \
    --input "ict_pp/csv/2012-2020_ICT_DSE.csv" \
    --sampling-n 25 \
    --n-permutations 4 \
    --num-workers 8

# Example: Test multiple models on the same dataset
python eval_sampling_bias.py --model gemma3:1b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
python eval_sampling_bias.py --model gemma3:4b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
python eval_sampling_bias.py --model gemma3:12b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
python eval_sampling_bias.py --model mistral:latest --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25

# Test on MMLU datasets
python eval_sampling_bias.py --model llama3.2:latest --input "mmlu/data/college_cs.csv" --sampling-n 25
python eval_sampling_bias.py --model llama3.2:latest --input "mmlu/data/sociology.csv" --sampling-n 25
```

**Parameters:**
- `--model`: Ollama model name
- `--input`: Path to MCQ CSV file
- `--sampling-n`: Number of samples per permutation (default: 15, recommended: 25)
- `--n-permutations`: Number of answer position permutations (default: 4)
- `--sampling-temp`: Temperature for sampling (default: 0.7)
- `--num-workers`: Parallel workers for sampling (default: 8)
- `--max-questions`: Limit number of questions (optional, for testing)

**Output:** `results/csv_results/{dataset}-{model}_sampling_n{N}.csv`

### Step 2: Run Statistical Analysis

Process all collected CSV files to generate statistics and analysis reports:

```bash
python batch_stat_analysis.py
```

This will:
- Scan `results/csv_results/` for all CSV files
- Generate analysis reports in `results/stat_results/{dataset}/`
- Create pairwise comparison heatmaps for each dataset
- Output text files with chi-square tests, bias scores, and recall metrics

**Output:**
- `results/stat_results/{dataset}/{dataset}-{model}_analysis.txt`
- `results/stat_results/{dataset}/{dataset}_pairwise_heatmap.png`

### Step 3: Apply PriDe Debiasing (Batch)

Run PriDe debiasing on all datasets and generate comparison visualizations:

```bash
python pride_batch_summary.py
```

This will:
- Process all CSV files in `results/csv_results/`
- Test alpha values from 0.0 to 1.0 to find optimal debiasing strength
- Generate comprehensive comparison plots grouped by dataset
- Create individual model-dataset visualizations
- Output summary report with all metrics

**Output:**
- `results/pride_summary/by_dataset/{dataset}/` - Dataset-level comparisons
  - `{dataset}_accuracy_comparison.png`
  - `{dataset}_bias_comparison.png`
  - `{dataset}_distribution_comparison.png`
  - `{dataset}_accuracy_by_position.png`
- `results/pride_summary/by_model_dataset/{dataset}-{model}/` - Individual plots
  - `accuracy_comparison.png`
  - `bias_metrics.png`
  - `distribution.png`
  - `summary.png`
- `results/pride_summary/SUMMARY_REPORT.txt` - Overall summary

### Step 4: Detailed Visualization (Optional)

For in-depth analysis of a specific model-dataset pair:

```bash
python pride_detail_eval.py "results/csv_results/2012-2020_ICT_DSE-gemma3_1b_sampling_n25.csv" --calibration-ratio 0.10
```

**Parameters:**
- First argument: Path to specific CSV file
- `--calibration-ratio`: Ratio of data for calibration (default: 0.10 = 10%)

**Output:** `results/pride_visualizations/{dataset}-{model}/`
- `SUMMARY_DASHBOARD.png` - Comprehensive overview
- `01_before_after_comparison/` - Radar charts and comparisons
- `02_choice_distributions/` - Choice distribution analysis
- `03_accuracy_analysis/` - Accuracy by position plots
- `04_alpha_selection/` - Alpha parameter selection analysis
- `05_detailed_metrics/` - Additional detailed metrics
- `analysis_report.txt` - Comprehensive text report

## Example Complete Workflow

```bash
# 1. Collect data for multiple models on ICT dataset
python eval_sampling_bias.py --model gemma3:1b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
python eval_sampling_bias.py --model gemma3:4b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25
python eval_sampling_bias.py --model gemma3:12b --input "ict_pp/csv/2012-2020_ICT_DSE.csv" --sampling-n 25

# 2. Run statistical analysis on all collected data
python batch_stat_analysis.py

# 3. Apply PriDe debiasing and generate visualizations
python pride_batch_summary.py

# 4. (Optional) Generate detailed visualization for specific model
python pride_detail_eval.py "results/csv_results/2012-2020_ICT_DSE-gemma3_1b_sampling_n25.csv"
```

## Understanding the Results

### Key Metrics

1. **Position Bias Score (σ)**: Standard deviation of choice percentages. Lower is better (more uniform). **σ ≥ 5 is considered significant bias**.

2. **Recall Std (RStd)**: Standard deviation of recall rates across positions. Measures performance balance. Lower indicates more balanced performance.

3. **Chi-square Tests**:
   - **GoF (Goodness-of-Fit)**: Tests if choice distribution deviates from uniform. Lower χ² and higher p-value are better.
   - **Independence**: Tests if accuracy depends on answer position. Higher p-value indicates position-independence.

4. **Overall Accuracy**: Proportion of correct answers. PriDe aims to improve or maintain this while reducing bias.

### Interpreting Visualizations

- **Red bars/points**: Baseline performance (before debiasing)
- **Blue/teal bars/points**: After PriDe debiasing
- **Dashed red line at 25%**: Uniform distribution reference
- **Green improvements / Red deteriorations**: Color-coded metric changes

## Requirements

See `requirements.txt` for Python dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- requests

## Link to Overleaf

https://www.overleaf.com/project/68be8d02fb134c31cf86398d
