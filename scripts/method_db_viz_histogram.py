#!/usr/bin/env python3
"""
Script to process similarity scores from experimental data and create visualizations.
Extracts dataset, retriever type, and attack method (aggd/hotflip) from folder structure.
Creates boolean column for attack vs clean classification.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import List, Dict
import argparse

# FILES_PATTERN = r"sample_(\d+)_self_similarity_scores_truncated_(start|end)\.json"
FILES_PATTERN = r"sample_(\d+)_self_similarity_scores_window.json"

# FILES_START = "test_method_"
FILES_START = "test_method_window20"


class SimilarityScoreAnalyzer:
    def __init__(self, base_directory: str):
        """
        Initialize the analyzer with base directory.

        Args:
            base_directory: Path to the directory containing FILES_START folders
        """
        self.base_directory = Path(base_directory)
        self.data = []

    def find_test_method_folders(self) -> List[Path]:
        """Find all folders that might contain experiment data."""
        folders = []
        for folder in self.base_directory.iterdir():
            if folder.is_dir() and folder.name.startswith(FILES_START):
                folders.append(folder)
        return sorted(folders)

    def find_similarity_files(self, folder: Path) -> List[Path]:
        """Find all similarity score files in folder and subfolders."""
        similarity_files = []
        pattern = re.compile(FILES_PATTERN)

        # Look in all subdirectories recursively
        for root, dirs, files in os.walk(folder):
            for file in files:
                if pattern.match(file):
                    similarity_files.append(Path(root) / file)

        return similarity_files

    def parse_filename(self, filepath: Path) -> Dict[str, str]:
        """Extract information from filename."""
        filename = filepath.name
        match = re.match(FILES_PATTERN, filename)

        if match:
            folder_info = self._extract_folder_info(filepath)
            return {
                "sample_number": int(match.group(1)),
                # "truncation_type": match.group(2),
                **folder_info,
            }
        return {}

    def _extract_folder_info(self, filepath: Path) -> Dict[str, str]:
        """Extract dataset, retriever type, and attack type from folder path."""
        parts = filepath.parts

        # Initialize default values
        dataset = "unknown"
        retriever_type = "unknown"
        attack_type = "unknown"
        score_function = "unknown"

        # Look through all path parts to extract information
        for part in parts:
            part_lower = part.lower()

            # Extract attack type (aggd or hotflip)
            for at in ["blackbox_hotflip", "aggd", "hotflip"]:
                if at in part_lower:
                    attack_type = at
                    break

            # Extract common dataset names (you may need to adjust these)
            for dataset_name in [
                "msmarco",
                "nq",
                "mirage",
                "hotpotqa",
            ]:
                if dataset_name in part_lower:
                    dataset = dataset_name

            # Extract retriever types (you may need to adjust these)
            for retriever in [
                "contriever",
                "contriever-ms",
                "ance",
                "minilm",
                "mpnet",
                "roberta",
            ]:
                if retriever in part_lower:
                    retriever_type = retriever

            for score in ["cos_sim", "dot"]:
                if score in part_lower:
                    score_function = score
                    break

        return {
            "dataset": dataset,
            "retriever_type": retriever_type,
            "attack_type": attack_type,
            "score_function": score_function,
        }

    def parse_token_length(self, token_info: str) -> int:
        """Extract token length from string like 'token length: 64'."""
        match = re.search(r"token length: (\d+)", token_info)
        return int(match.group(1)) if match else 0

    def process_json_file(self, filepath: Path) -> List[Dict]:
        """Process a single JSON file and extract data."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            file_info = self.parse_filename(filepath)
            processed_entries = []

            for entry in data:
                # Each entry: [index_info, similarity_values, attack_or_clean, token_length_info]
                if len(entry) >= 4:
                    index_info = entry[0]
                    similarity_values = entry[1]
                    attack_or_clean = entry[2]  # 'attack' or 'clean'
                    token_length_info = entry[3]

                    # Extract top k index
                    top_k_match = re.search(r"top k index: (\d+)", index_info)
                    top_k_index = int(top_k_match.group(1)) if top_k_match else 0

                    # Extract token length
                    token_length = self.parse_token_length(token_length_info)

                    # Create boolean for attack (1 if attack, 0 if clean)
                    is_attack = 1 if attack_or_clean.lower() == "attack" else 0

                    # Calculate statistics
                    similarity_array = np.array(similarity_values)

                    entry_data = {
                        **file_info,
                        "top_k_index": top_k_index,
                        "similarity_values": similarity_values,
                        "is_attack": is_attack,
                        "attack_or_clean": attack_or_clean,  # Keep original for reference
                        "token_length": token_length,
                        "num_values": len(similarity_values),
                        "mean_similarity": float(np.mean(similarity_array)),
                        "std_similarity": float(np.std(similarity_array)),
                        "min_similarity": float(np.min(similarity_array)),
                        "max_similarity": float(np.max(similarity_array)),
                        "filepath": str(filepath),
                    }

                    processed_entries.append(entry_data)

            return processed_entries

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return []

    def process_all_files(self):
        """Process all similarity score files."""
        folders = self.find_test_method_folders()
        print(f"Found {len(folders)} folders to process")

        total_files = 0
        for folder in folders:
            print(f"\nProcessing: {folder.name}")
            similarity_files = self.find_similarity_files(folder)
            print(f"  Found {len(similarity_files)} similarity files")

            for file_path in similarity_files:
                entries = self.process_json_file(file_path)
                self.data.extend(entries)
                total_files += 1

        print(f"\nTotal files processed: {total_files}")
        print(f"Total data entries: {len(self.data)}")

        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.data)
        return self.df

    def create_attack_comparison_plots(self, method, save_plots: bool = True):
        """
        Create comparison plots split by score function into separate PNG figures.
        Each figure contains subplots grouped by retriever and dataset.
        Each plot shows 3 lines: clean, hotflip, and aggd with std deviation bands.
        X-axis represents number of removed tokens.
        """
        if self.df.empty:
            print("No data to visualize. Run process_all_files() first.")
            return

        # Define grouping columns - adjust these based on your actual column names
        group_columns = []
        if "retriever_type" in self.df.columns:
            group_columns.append("retriever_type")
        if "dataset" in self.df.columns:
            group_columns.append("dataset")

        # Check if score_function column exists
        if "score_function" not in self.df.columns:
            print("Warning: No 'score_function' column found. Creating single plot.")
            self._create_single_attack_comparison_plot(self.df, "Overall", save_plots)
            return

        # Group by score function first
        score_function_groups = self.df.groupby("score_function")

        for score_func, score_func_data in score_function_groups:
            print(f"Creating plots for score function: {score_func}")

            if not group_columns:
                # If no other grouping columns, create single plot for this score function
                fig, ax = plt.subplots(figsize=(10, 6))
                self._create_single_attack_comparison_plot(
                    score_func_data, f"{score_func}", ax
                )

                if save_plots:
                    filename = f"plots/attack_comparison_{score_func}_{method}.png"
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    print(f"Saved attack comparison plot as '{filename}'")

                plt.show()
                continue

            # Group the data by remaining columns (retriever_type, dataset, etc.)
            grouped = score_func_data.groupby(group_columns)
            n_groups = len(grouped)

            if n_groups == 0:
                continue

            # Calculate subplot grid dimensions
            n_cols = min(3, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Handle different subplot configurations
            if n_groups == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            plot_idx = 0
            for group_name, group_data in grouped:
                if plot_idx >= len(axes):
                    break
                print(f"token length in group {group_name}_{score_func}:")
                print(group_data["token_length"].describe())
                ax = axes[plot_idx]
                self._create_single_attack_comparison_plot(group_data, group_name, ax)
                plot_idx += 1

            # Hide empty subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)

            # Add overall title for the score function
            fig.suptitle(
                f"Attack Comparison - Score Function: {score_func}", fontsize=16, y=0.98
            )

            plt.tight_layout()

            if save_plots:
                # Clean the score function name for filename
                score_func_clean = (
                    str(score_func)
                    .replace("/", "_")
                    .replace(" ", "_")
                    .replace(".", "_")
                )
                filename = (
                    f"plots/attack_comparison_{score_func_clean}_{method}_histogram.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved attack comparison plots as '{filename}'")

            plt.show()

    def _create_single_attack_comparison_plot(self, data, group_name, ax=None):
        """
        Create side-by-side histograms (bar plots) of the difference between max and min similarity values for each attack type and clean data.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "clean": "#2E8B57",  # Sea Green
            "hotflip": "#DC143C",  # Crimson
            "aggd": "#4169E1",  # Royal Blue
            "blackbox_hotflip": "orange",
        }

        # Separate data by type
        data_types = {
            "Clean": data[data["attack_or_clean"] == "clean"],
            "HotFlip": (
                data[data["attack_type"] == "hotflip"]
                if "attack_type" in data.columns
                else data[data["attack_or_clean"] == "hotflip"]
            ),
            "AGGD": (
                data[data["attack_type"] == "aggd"]
                if "attack_type" in data.columns
                else data[data["attack_or_clean"] == "aggd"]
            ),
            "Blackbox HotFlip": (
                data[data["attack_type"] == "blackbox_hotflip"]
                if "attack_type" in data.columns
                else data[data["attack_or_clean"] == "blackbox_hotflip"]
            ),
        }

        bins = 30
        all_diffs = []
        for d in data_types.values():
            if not d.empty:
                all_diffs.extend(
                    d["similarity_values"]
                    .apply(lambda vals: np.max(vals) - np.min(vals))
                    .tolist()
                )
        if not all_diffs:
            print(f"No data found for any type in group {group_name}")
            return

        # Use common bins for all types
        min_diff = min(all_diffs)
        max_diff = max(all_diffs)
        bin_edges = np.linspace(min_diff, max_diff, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = (bin_edges[1] - bin_edges[0]) * 0.8  # bar width

        # Plot each type side by side
        n_types = sum(1 for d in data_types.values() if not d.empty)
        offset_range = np.linspace(-width * n_types / 2, width * n_types / 2, n_types)
        idx = 0
        for label, d in data_types.items():
            if d.empty:
                continue
            diffs = (
                d["similarity_values"]
                .apply(lambda vals: np.max(vals) - np.min(vals))
                .values
            )
            counts, _ = np.histogram(diffs, bins=bin_edges)
            ax.bar(
                bin_centers + offset_range[idx],
                counts,
                width=width / n_types,
                label=label,
                color=colors.get(label.lower().replace(" ", ""), "gray"),
                alpha=0.8,
                align="center",
                edgecolor="black",
            )
            idx += 1

        ax.set_xlabel("Max-Min Similarity Difference")
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of Max-Min Similarity Differences: {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_comprehensive_attack_analysis(self, method, save_plots: bool = True):
        """`
        Create a comprehensive analysis including both overview and detailed comparisons.
        """
        print("Creating comprehensive attack analysis...")

        # Create the new attack comparison plots
        self.create_attack_comparison_plots(method, save_plots)

        # Create summary statistics
        self._print_attack_summary_stats()

    def _print_attack_summary_stats(self):
        """
        Print summary statistics for attack effectiveness.
        """
        print("\n" + "=" * 60)
        print("ATTACK EFFECTIVENESS SUMMARY")
        print("=" * 60)

        # Overall statistics
        if "attack_or_clean" in self.df.columns:
            overall_stats = self.df.groupby("attack_or_clean")["mean_similarity"].agg(
                ["mean", "std", "count"]
            )
            print("\nOverall Mean Similarity by Type:")
            print(overall_stats)

        # Attack type comparison
        if "attack_type" in self.df.columns:
            attack_data = self.df[self.df["is_attack"] == 1]
            if not attack_data.empty:
                attack_stats = attack_data.groupby("attack_type")[
                    "mean_similarity"
                ].agg(["mean", "std", "count"])
                print("\nAttack Type Effectiveness:")
                print(attack_stats)

        # Group-wise analysis
        group_columns = []
        if "retriever_type" in self.df.columns:
            group_columns.append("retriever_type")
        if "dataset" in self.df.columns:
            group_columns.append("dataset")

        if group_columns:
            print(f"\nMean Similarity by {' + '.join(group_columns)}:")
            group_stats = (
                self.df.groupby(group_columns + ["attack_or_clean"])["mean_similarity"]
                .mean()
                .unstack()
            )
            print(group_stats)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze similarity scores from experimental data"
    )
    parser.add_argument(
        "--directory",
        default="./data_cache/outputs",
        help="Base directory containing experiment folders",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--save-plots", action="store_true", default=True, help="Save plots to files"
    )
    parser.add_argument("--method", default="truncation", help="drift method name")
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SimilarityScoreAnalyzer(args.directory)

    # Process all files
    print("Processing similarity score files...")
    df = analyzer.process_all_files()

    if df.empty:
        print("No data found. Check your directory structure and file patterns.")
        return

    # Create visualizations
    if not args.no_plots:
        analyzer.create_comprehensive_attack_analysis(
            method=args.method, save_plots=args.save_plots
        )

    # Optionally save processed data
    output_file = "processed_similarity_data_histogram.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")


if __name__ == "__main__":
    main()
