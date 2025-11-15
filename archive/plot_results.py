"""
Generates a bar chart comparing model accuracy before and after PriDe debiasing.
Handles multiple datasets by creating a separate plot for each.
"""

import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def parse_report(file_path: Path) -> dict:
    """Extracts key metrics from a PriDe report file."""
    text = file_path.read_text()
    
    # Generalize extraction of dataset and model name
    # This regex captures the dataset and model parts of the filename.
    match = re.search(r'^(.*?)-((?:gemma|llama|mistral).*?)_sampling', file_path.name)
    if not match:
        return None
        
    dataset_name = match.group(1)
    model_name = match.group(2)
    
    # Extract accuracies
    original_acc_match = re.search(r"Original Accuracy \(from argmax\):\s*([0-9.]+)", text)
    debiased_acc_match = re.search(r"Debiased Accuracy:\s*([0-9.]+)", text)
    
    if not original_acc_match or not debiased_acc_match:
        return None
        
    return {
        "dataset": dataset_name,
        "model": model_name.replace('_', ' ').replace('-', ' '), # Clean up model name for display
        "original_accuracy": float(original_acc_match.group(1)),
        "debiased_accuracy": float(debiased_acc_match.group(1)),
    }

def plot_accuracy_comparison(results: list, dataset_name: str):
    """Generates and saves a grouped bar chart of the results for a specific dataset."""
    
    # Convert data into a long-form DataFrame suitable for seaborn
    plot_data = []
    for res in results:
        plot_data.append({"Model": res["model"], "Accuracy Type": "Original", "Accuracy": res["original_accuracy"]})
        plot_data.append({"Model": res["model"], "Accuracy Type": "Debiased (PriDe)", "Accuracy": res["debiased_accuracy"]})
        
    df = pd.DataFrame(plot_data)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Accuracy Type", ax=ax, palette="viridis")
    
    # Add labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)
        
    # Formatting
    ax.set_title(f"Effectiveness of PriDe Debiasing on Model Accuracy\nDataset: {dataset_name.replace('_', ' ')}", fontsize=18, weight='bold')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, max(1.0, df['Accuracy'].max() * 1.15))
    ax.legend(title="Accuracy Type", loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure with a dynamic name
    output_path = f"pride_effectiveness_comparison_{dataset_name}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved successfully to: {output_path}")

def main():
    """Main function to find reports, parse them, and generate the plot."""
    results_dir = Path("results") / "pride_optimized_stat_results"
    
    if not results_dir.exists():
        print(f"Error: Directory not found at '{results_dir}'")
        return
        
    all_results = []
    report_files = list(results_dir.glob("*_pride_report.txt"))
    
    if not report_files:
        print(f"No PriDe report files found in '{results_dir}'.")
        return

    for report_file in report_files:
        result = parse_report(report_file)
        if result:
            all_results.append(result)
            
    if not all_results:
        print("Could not parse any results from the report files.")
        return

    # Group results by dataset
    results_by_dataset = defaultdict(list)
    for res in all_results:
        results_by_dataset[res['dataset']].append(res)

    # Generate a plot for each dataset
    for dataset, results in results_by_dataset.items():
        print(f"\n---> Generating plot for dataset: {dataset}")
        # Sort results by model name for consistent plotting
        results.sort(key=lambda x: x['model'])
        plot_accuracy_comparison(results, dataset)

if __name__ == "__main__":
    main()
