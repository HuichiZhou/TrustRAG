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

DRIFT_OPTIONS = {
    "truncation": {
        "FILES_PATTERN": r"sample_(\d+)_self_similarity_scores_truncated_(start|end)\.json",
    },
    "window": {
        "FILES_PATTERN": r"sample_(\d+)_self_similarity_scores_window.json",
    },
    "uniform": {
        "FILES_PATTERN": r"sample_(\d+)_self_similarity_scores_uniform.json",
    },
}

FILES_START = "test_method_"


class SimilarityScoreAnalyzer:
    def __init__(self, base_directory: str, drift: str):
        """
        Initialize the analyzer with base directory.

        Args:
            base_directory: Path to the directory containing FILES_START folders
        """
        drift_config = DRIFT_OPTIONS[drift]
        files_pattern = drift_config["FILES_PATTERN"]
        files_start = FILES_START
        self.drift = drift
        self.base_directory = Path(base_directory)
        self.files_pattern = files_pattern
        self.files_start = files_start
        self.data = []

    def find_test_method_folders(self) -> List[Path]:
        """Find all folders that might contain experiment data."""
        folders = []
        for folder in self.base_directory.iterdir():
            if folder.is_dir() and folder.name.startswith(self.files_start):
                folders.append(folder)
        return sorted(folders)

    def find_similarity_files(self, folder: Path) -> List[Path]:
        """Find all similarity score files in folder and subfolders."""
        similarity_files = []
        pattern = re.compile(self.files_pattern)

        # Look in all subdirectories recursively
        for root, dirs, files in os.walk(folder):
            for file in files:
                if pattern.match(file):
                    similarity_files.append(Path(root) / file)

        return similarity_files

    def parse_filename(self, filepath: Path) -> Dict[str, str]:
        """Extract information from filename."""
        filename = filepath.name
        match = re.match(self.files_pattern, filename)

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
        num_tokens_removed = None

        # Look through all path parts to extract information
        for part in parts:
            part_lower = part.lower()

            # Extract attack type (aggd or hotflip)
            for at in ["aggd", "hotflip", "blackbox_hotflip"]:
                if at in part_lower:
                    attack_type = at
                    break

            # Extract common dataset names
            for dataset_name in ["msmarco", "nq", "mirage", "hotpotqa"]:
                if dataset_name in part_lower:
                    dataset = dataset_name

            # Extract retriever types
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

            # Extract num_tokens_removed from folder name
            match = re.search(r"num_tokens_removed_(\d+)", part_lower)
            if match:
                num_tokens_removed = int(match.group(1))

        return {
            "dataset": dataset,
            "retriever_type": retriever_type,
            "attack_type": attack_type,
            "score_function": score_function,
            "num_tokens_removed": num_tokens_removed,
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
                        "final_similarity": (
                            similarity_values[-1] if similarity_values else 0
                        ),
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

    def create_attack_comparison_plots(self, save_plots: bool = True):
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
                    score_func_data, f"{score_func}", False, ax
                )

                if save_plots:
                    filename = f"plots/attack_comparison_{score_func}.png"
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
                axes = np.array([axes])
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
                self._create_single_attack_comparison_plot(
                    group_data, group_name, False, ax
                )
                plot_idx += 1

            # Hide empty subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)

            # Add overall title for the score function
            fig.suptitle(
                f"Attack Comparison - Score Function: {score_func} - Drift: {self.drift}",
                fontsize=16,
                y=0.98,
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
                    f"plots/attack_comparison_{score_func_clean}_{self.drift}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved attack comparison plots as '{filename}'")

            plt.show()

    def _create_single_attack_comparison_plot(
        self, data, group_name, save_individual=False, ax=None
    ):
        """
        Create a comparison plot for clean vs hotflip vs aggd.
        For truncation: x-axis is the index in the similarity list (tokens removed).
        For window/uniform: x-axis is num_tokens_removed from folder info.
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
        attack_types = [
            ("Clean", data[data["attack_or_clean"] == "clean"], colors["clean"]),
            (
                "HotFlip",
                data[data.get("attack_type", data["attack_or_clean"]) == "hotflip"],
                colors["hotflip"],
            ),
            (
                "AGGD",
                data[data.get("attack_type", data["attack_or_clean"]) == "aggd"],
                colors["aggd"],
            ),
            (
                "Blackbox HotFlip",
                data[
                    data.get("attack_type", data["attack_or_clean"])
                    == "blackbox_hotflip"
                ],
                colors["blackbox_hotflip"],
            ),
        ]

        if self.drift == "truncation":
            # Plot as before: x = index in similarity_values
            for label, d, color in attack_types:
                if d.empty:
                    continue
                sim_vals = np.array(d["similarity_values"].tolist())
                if sim_vals.size == 0:
                    continue
                max_len = max(len(vals) for vals in d["similarity_values"])
                x_values = np.arange(max_len)
                # Pad with nan for unequal lengths
                sim_vals_padded = np.full((len(sim_vals), max_len), np.nan)
                for i, row in enumerate(sim_vals):
                    sim_vals_padded[i, : len(row)] = row
                y_values = np.nanmean(sim_vals_padded, axis=0)
                y_stds = np.nanstd(sim_vals_padded, axis=0)
                ax.plot(
                    x_values, y_values, color=color, linewidth=2, label=label, alpha=0.8
                )
                ax.fill_between(
                    x_values,
                    y_values - y_stds,
                    y_values + y_stds,
                    color=color,
                    alpha=0.2,
                )
            ax.set_xlabel("Number of Removed Tokens")
            ax.set_ylabel("Mean Similarity Score")
        else:
            # For window/uniform: x = num_tokens_removed, y = final similarity value
            for label, d, color in attack_types:
                if d.empty or "num_tokens_removed" not in d.columns:
                    continue
                # Group by num_tokens_removed, aggregate final_similarity
                grouped = d.groupby("num_tokens_removed")["final_similarity"]
                x = []
                y = []
                yerr = []
                for num_removed, vals in grouped:
                    x.append(num_removed)
                    y.append(np.mean(vals))
                    yerr.append(np.std(vals))
                if not x:
                    continue
                x = np.array(x)
                y = np.array(y)
                yerr = np.array(yerr)
                sort_idx = np.argsort(x)
                ax.errorbar(
                    x[sort_idx],
                    y[sort_idx],
                    yerr=yerr[sort_idx],
                    fmt="-o",
                    color=color,
                    label=label,
                    alpha=0.8,
                )
            ax.set_xlabel("Number of Tokens Removed")
            ax.set_ylabel("Final Similarity Score")

        ax.set_title(f"Attack Comparison: {group_name} - Drift: {self.drift}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_individual:
            group_name_clean = str(group_name).replace("/", "_").replace(" ", "_")
            plt.savefig(
                f"attack_comparison_{group_name_clean}_{self.drift}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Saved individual plot as 'attack_comparison_{group_name_clean}_{self.drift}.png'"
            )

    def create_comprehensive_attack_analysis(self, save_plots: bool = True):
        """
        Create a comprehensive analysis including both overview and detailed comparisons.
        """
        print("Creating comprehensive attack analysis...")

        # Create the new attack comparison plots
        self.create_attack_comparison_plots(save_plots)

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
    parser.add_argument(
        "--drift",
        choices=["truncation", "window", "uniform"],
        default="window",
        help="Drift method to analyze (truncation, window, uniform)",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SimilarityScoreAnalyzer(args.directory, args.drift)

    # Process all files
    print("Processing similarity score files...")
    df = analyzer.process_all_files()

    if df.empty:
        print("No data found. Check your directory structure and file patterns.")
        return

    # Create visualizations
    if not args.no_plots:
        analyzer.create_comprehensive_attack_analysis(save_plots=args.save_plots)

    # Optionally save processed data
    output_file = f"processed_similarity_data_{args.drift}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")


if __name__ == "__main__":
    main()
