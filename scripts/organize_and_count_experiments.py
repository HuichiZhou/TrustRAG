#!/usr/bin/env python3
"""
TrustRAG Experiment Counter and Organizer
Analyzes log files to count different experimental configurations
"""

import re
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd


class TrustRAGExperimentAnalyzer:
    def __init__(self, logs_directory):
        self.logs_dir = Path(logs_directory)
        self.experiments = []
        self.analysis = {}

    def parse_folder_structure(self):
        """Parse only the folder structure (dataset-retriever-model combinations)"""
        folders = []

        for item in self.logs_dir.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                folders.append(item.name)

        return folders

    def parse_experiment_name(self, experiment_name):
        """Parse individual experiment log file names to extract parameters"""
        # Remove .log extension
        name = experiment_name.replace(".log", "")

        experiment_config = {}

        # Handle model extraction (including meta-llama and mistralai)
        # Look for specific model patterns in your naming convention
        if "gpt-4o" in name:
            experiment_config["model"] = "gpt-4o"
        elif "Llama-3.1" in name:
            experiment_config["model"] = "Llama-3.1"
        elif "Mistral-Nemo" in name:
            experiment_config["model"] = "Mistral-Nemo"
        else:
            # Fallback to regex approach
            model_match = re.search(
                r"model_(true_)?([^-]+(?:-[^-]+)*?)(?:-M10x|-attack_|$)", name
            )
            if model_match:
                prefix = model_match.group(1) or ""  # 'true_' or empty
                model_name = model_match.group(2)
                experiment_config["model"] = prefix + model_name

        # Split by dashes and extract other components
        parts = name.split("-")

        for part in parts:
            if part.startswith("attack_"):
                experiment_config["attack"] = part.replace("attack_", "")
            elif part.startswith("removal_"):
                experiment_config["removal"] = part.replace("removal_", "")
            elif part.startswith("defend_"):
                experiment_config["defense"] = part.replace("defend_", "")
            elif part.startswith("adv_per_query"):
                experiment_config["adv_per_query"] = part.replace("adv_per_query", "")
            elif part.startswith("adv_a_position_"):
                experiment_config["adv_position"] = part.replace("adv_a_position_", "")
            elif part.startswith("Top_"):
                experiment_config["top_k"] = part.replace("Top_", "")
            elif part.startswith("Seed_"):
                experiment_config["seed"] = part.replace("Seed_", "")
            elif part in ["cos_sim", "dot"]:
                experiment_config["similarity_metric"] = part
            elif "M10xRepeat10" in part:
                experiment_config["experimental_setup"] = part
            elif "no_questions_both_sides" in part:
                experiment_config["question_mode"] = part

        return experiment_config

    def check_duplicates_in_folder(self, folder_path, log_files):
        """Check for duplicate experiments within a single folder"""
        file_configs = []
        duplicates_found = False

        for log_file in log_files:
            config = self.parse_experiment_name(log_file.name)
            # Create a signature for this experiment (excluding filename-specific parts)
            signature = tuple(
                sorted(
                    [
                        (k, v)
                        for k, v in config.items()
                        if k not in ["folder"]  # Exclude metadata we add
                    ]
                )
            )
            file_configs.append((log_file.name, signature, config))

        # Group by signature to find duplicates
        signature_groups = defaultdict(list)
        for filename, signature, config in file_configs:
            signature_groups[signature].append(filename)

        # Report duplicates
        for signature, filenames in signature_groups.items():
            if len(filenames) > 1:
                duplicates_found = True
                print(
                    f"⚠️  WARNING: Duplicate experiments found in folder {folder_path.name}:"
                )
                print(
                    f"   Configuration: {dict(signature) if signature else 'Unable to parse'}"
                )
                print(f"   Files: {', '.join(filenames)}")
                print()

        return duplicates_found, len(signature_groups)

    def analyze_experiments(self):
        """Analyze only .log files inside folders and check for duplicates"""
        folders = self.parse_folder_structure()

        print(f"Found {len(folders)} experiment folders")
        print("Ignoring job_*.txt files - analyzing only .log files inside folders")
        print("Checking for duplicate experiments within each folder...\n")

        # Parse individual log files within folders only
        all_experiments = []
        folder_summary = []
        total_duplicates_found = False

        for folder in folders:
            folder_path = self.logs_dir / folder
            if folder_path.exists():
                log_files = list(folder_path.glob("*.log"))

                # Extract dataset and retriever from folder name
                folder_dataset = None
                folder_retriever = None

                # Extract dataset from folder name
                folder_dataset_match = re.search(r"dataset_([^-\s]+)", folder)
                if folder_dataset_match:
                    folder_dataset = folder_dataset_match.group(1)

                # Extract retriever from folder name
                folder_retriever_match = re.search(
                    r"retriver_([^-]+(?:-[^-]+)*?)(?:-|$)", folder
                )
                if folder_retriever_match:
                    folder_retriever = folder_retriever_match.group(1)

                # Check for duplicates in this folder
                duplicates_found, unique_configs = self.check_duplicates_in_folder(
                    folder_path, log_files
                )
                if duplicates_found:
                    total_duplicates_found = True

                folder_summary.append(
                    (folder, len(log_files), unique_configs, duplicates_found)
                )
                print(
                    f"Folder {folder}: {len(log_files)} .log files, {unique_configs} unique configurations"
                    + (" ⚠️  HAS DUPLICATES" if duplicates_found else " ✅")
                )

                for log_file in log_files:
                    config = self.parse_experiment_name(log_file.name)
                    config["folder"] = folder  # Add folder info for tracking
                    config["filename"] = log_file.name  # Add filename for tracking

                    # Use folder-level dataset/retriever if not found in filename
                    config["dataset"] = folder_dataset
                    config["retriever"] = folder_retriever

                    all_experiments.append(config)

        print(f"\nTotal .log files found: {len(all_experiments)}")

        if total_duplicates_found:
            print("\n🔶 DUPLICATE DETECTION SUMMARY:")
            print("   Some folders contain duplicate experiment configurations.")
            print("   See warnings above for details. No files were modified.")
        else:
            print(
                "\n✅ NO DUPLICATES FOUND: All experiments are unique within their folders."
            )

        print("-" * 60)

        self.experiments = all_experiments
        self.folder_summary = folder_summary
        return self.count_configurations()

    def count_configurations(self):
        """Count different types of experimental configurations"""
        if not self.experiments:
            return {}

        # Count each parameter
        counts = {}
        for param in [
            "dataset",
            "retriever",
            "model",
            "attack",
            "removal",
            "defense",
            "similarity_metric",
            "adv_per_query",
            "adv_position",
            "top_k",
            "seed",
        ]:
            values = [
                exp.get(param, "unknown") for exp in self.experiments if exp.get(param)
            ]
            counts[param] = Counter(values)

        # Count combinations
        dataset_retriever_model = [
            (exp.get("dataset", ""), exp.get("retriever", ""), exp.get("model", ""))
            for exp in self.experiments
        ]
        counts["dataset_retriever_model_combinations"] = Counter(
            dataset_retriever_model
        )

        attack_configs = [
            (
                exp.get("attack", ""),
                exp.get("removal", ""),
                exp.get("similarity_metric", ""),
                exp.get("adv_per_query", ""),
                exp.get("adv_position", ""),
            )
            for exp in self.experiments
        ]
        counts["attack_configurations"] = Counter(attack_configs)

        self.analysis = counts
        return counts

    def generate_summary(self):
        """Generate a comprehensive summary of .log file experiments only"""
        if not self.analysis:
            self.analyze_experiments()

        summary = []
        summary.append("=== TrustRAG Experiment Summary (.log files only) ===\n")

        # Folder breakdown
        summary.append("FOLDER BREAKDOWN:")
        duplicates_summary = []
        for folder, file_count, unique_count, has_duplicates in self.folder_summary:
            status = " ⚠️  DUPLICATES" if has_duplicates else " ✅"
            summary.append(
                f"  {folder}: {file_count} files, {unique_count} unique configs{status}"
            )
            if has_duplicates:
                duplicates_summary.append(folder)

        summary.append(
            f"\nTotal .log files across all folders: {len(self.experiments)}"
        )

        if duplicates_summary:
            summary.append(f"Folders with duplicates: {len(duplicates_summary)}")
            summary.append("⚠️  Note: Some experiments may be duplicated within folders")
        else:
            summary.append("✅ All experiments are unique within their folders")

        summary.append("")

        # Parameter counts
        for param, counter in self.analysis.items():
            if param in [
                "dataset_retriever_model_combinations",
                "attack_configurations",
            ]:
                continue
            summary.append(f"{param.upper()} ({len(counter)} unique values):")
            for value, count in counter.most_common():
                summary.append(f"  {value}: {count}")
            summary.append("")

        # Combinations
        summary.append("DATASET-RETRIEVER-MODEL COMBINATIONS:")
        for combo, count in self.analysis[
            "dataset_retriever_model_combinations"
        ].most_common():
            dataset, retriever, model = combo
            summary.append(f"  {dataset}-{retriever}-{model}: {count}")
        summary.append("")

        summary.append("ATTACK CONFIGURATIONS:")
        for combo, count in self.analysis["attack_configurations"].most_common():
            attack, removal, sim_metric, adv_per_query, adv_pos = combo
            summary.append(
                f"  {attack}/{removal}/{sim_metric}/q{adv_per_query}/{adv_pos}: {count}"
            )

        return "\n".join(summary)

    def export_to_csv(self, output_file="trustrag_log_experiments.csv"):
        """Export .log file experiment details to CSV for further analysis"""
        if not self.experiments:
            self.analyze_experiments()

        df = pd.DataFrame(self.experiments)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} .log file experiments to {output_file}")
        return df


# Usage example:
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrustRAGExperimentAnalyzer("logs")

    # Run analysis
    counts = analyzer.analyze_experiments()

    # Print summary
    print(analyzer.generate_summary())

    # Export to CSV
    df = analyzer.export_to_csv()

    # Additional analysis - missing experiments
    print("\n=== MISSING EXPERIMENT DETECTION ===")

    # Expected combinations (based on your folder structure)
    datasets = ["hotpotqa", "mirage", "msmarco", "nq"]
    retrievers = ["ance", "contriever", "contriever-ms", "minilm", "roberta", "mpnet"]
    models = ["gpt-4o", "Llama-3.1", "Mistralai-Nemo"]

    expected_base_combinations = len(datasets) * len(retrievers) * len(models)
    print(f"Expected base combinations: {expected_base_combinations}")

    actual_combinations = len(analyzer.analysis["dataset_retriever_model_combinations"])
    print(f"Actual combinations found: {actual_combinations}")

    if actual_combinations < expected_base_combinations:
        print("Some base combinations might be missing!")
