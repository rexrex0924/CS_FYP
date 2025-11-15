"""
Extract MMLU dataset from HuggingFace and convert to CSV format.

This script downloads a specific category from the cais/mmlu dataset
and converts it to the format required for positional bias evaluation.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def extract_mmlu_category(category: str, output_dir: str = "./mmlu/data"):
    """
    Extract a specific MMLU category and save as CSV.
    
    Args:
        category: MMLU category name (e.g., "high_school_mathematics", "professional_law")
        output_dir: Directory to save the output CSV file
    """
    print(f"Extracting MMLU category: {category}")
    
    # Load data from HuggingFace
    split = f"{category}/test-00000-of-00001.parquet"
    print(f"Loading from: hf://datasets/cais/mmlu/{split}")
    
    try:
        df = pd.read_parquet("hf://datasets/cais/mmlu/" + split)
        print(f"Successfully loaded {len(df)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNote: You may need to install pyarrow:")
        print("  pip install pyarrow")
        return
    
    # Display info
    print(f"\nDataset info:")
    print(f"  Total questions: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Drop subject column if it exists
    if 'subject' in df.columns:
        df.drop(columns=['subject'], inplace=True)
    
    # Extract individual options from the choices list
    print("\nExtracting options A, B, C, D...")
    option_a = []
    option_b = []
    option_c = []
    option_d = []
    
    for choices in df['choices']:
        option_a.append(choices[0])
        option_b.append(choices[1])
        option_c.append(choices[2])
        option_d.append(choices[3])
    
    # Create ID column
    id_column = ['q' + str((index + 1)) for index in df.index]
    
    # Convert numeric answer (0-3) to letter (A-D)
    print("Converting answers to letter format...")
    answer = []
    for i in range(len(df)):
        if df['answer'].iloc[i] == 0:
            answer.append('A')
        elif df['answer'].iloc[i] == 1:
            answer.append('B')
        elif df['answer'].iloc[i] == 2:
            answer.append('C')
        else:
            answer.append('D')
    
    # Create formatted dataframe
    formatted_df = pd.DataFrame({
        'id': id_column,
        'question': df['question'].values,
        'option_a': option_a,
        'option_b': option_b,
        'option_c': option_c,
        'option_d': option_d,
        'answer': answer
    })
    
    print("\nFormatted dataset preview:")
    print(formatted_df.head())
    
    # Verify answer distribution
    answer_dist = formatted_df['answer'].value_counts().sort_index()
    print(f"\nAnswer distribution:")
    for ans, count in answer_dist.items():
        percentage = (count / len(formatted_df) * 100)
        print(f"  {ans}: {count} ({percentage:.1f}%)")
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use shorter name without "_dataset" suffix to match existing convention
    output_file = output_path / f"{category}.csv"
    
    formatted_df.to_csv(output_file, index=False)
    print(f"\n✅ Successfully saved to: {output_file}")
    print(f"   Total questions: {len(formatted_df)}")
    
    return formatted_df


def list_available_categories():
    """List some common MMLU categories."""
    categories = {
        "STEM": [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "electrical_engineering",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "machine_learning",
        ],
        "Humanities": [
            "formal_logic",
            "high_school_european_history",
            "high_school_us_history",
            "high_school_world_history",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "moral_disputes",
            "moral_scenarios",
            "philosophy",
            "prehistory",
            "professional_law",
            "world_religions",
        ],
        "Social Sciences": [
            "econometrics",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_microeconomics",
            "high_school_psychology",
            "human_sexuality",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
        ],
        "Other": [
            "business_ethics",
            "clinical_knowledge",
            "college_medicine",
            "global_facts",
            "human_aging",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "nutrition",
            "professional_accounting",
            "professional_medicine",
            "virology",
        ],
    }
    
    print("\n=== Available MMLU Categories ===\n")
    for domain, cats in categories.items():
        print(f"{domain}:")
        for cat in cats:
            print(f"  - {cat}")
        print()
    
    return categories


def main():
    parser = argparse.ArgumentParser(
        description="Extract MMLU dataset from HuggingFace and convert to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract high school mathematics
  python extract_mmlu_dataset.py --category high_school_mathematics

  # Extract professional law with custom output directory
  python extract_mmlu_dataset.py --category professional_law --output ./my_datasets

  # List all available categories
  python extract_mmlu_dataset.py --list-categories

Note: Requires pyarrow to be installed:
  pip install pyarrow
        """
    )
    
    parser.add_argument(
        "--category",
        type=str,
        help="MMLU category name (e.g., 'high_school_mathematics', 'professional_law')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mmlu/data",
        help="Output directory for CSV files (default: ./mmlu/data)"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available MMLU categories"
    )
    
    args = parser.parse_args()
    
    if args.list_categories:
        list_available_categories()
        return
    
    if not args.category:
        print("Error: --category is required (or use --list-categories to see available options)")
        parser.print_help()
        return
    
    try:
        extract_mmlu_category(args.category, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have pyarrow installed:")
        print("  pip install pyarrow")


if __name__ == "__main__":
    main()

