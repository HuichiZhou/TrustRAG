import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np  # For NaN values

REMOVAL_METHODS = [
    "kmeans_ngram",
    "drift",
    "perplexity",
    "drift_perplexity",
]  #     "kmeans_ngram", "drift", "drift_kmeans_ngram", "perplexity",

REMOVAL_COLORS = dict(
    zip(REMOVAL_METHODS, sns.color_palette("Set2", n_colors=len(REMOVAL_METHODS)))
)


def parse_log_file(file_path):
    """
    Extracts data from a single log file.
    """
    desired_patterns = {
        "success_injection_rate": r"Success injection rate in top 5 contents: (\d+\.\d+%)",
        "inference_time": r"Total run time for inference, removal and defense: (\d+\.\d+)",
        "correct_answer_pct": r"Correct Answer Percentage: (\d+\.\d+%)",
        "incorrect_answer_pct": r"Incorrect Answer Percentage: (\d+\.\d+%)",
        "false_removal_rate": r"False removal rate in true content contents: (\d+\.\d+%)",
        "success_removal_rate": r"Success removal rate in adv contents: (\d+\.\d+%)",
    }

    data = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        content = "".join(lines)

        # Extract experiment name
        experiment_name = None
        for line in lines:
            if "Experiment Name --" in line:
                match = re.search(r"Experiment Name --\s*(.*)", line)
                if match:
                    experiment_name = match.group(1).strip()
                break

        data["experiment_full_name"] = (
            experiment_name  # Keep full name for debugging/context
        )

        # Extract components from the experiment name
        if experiment_name:
            patterns = {
                "dataset": r"dataset_([^-\s]+)",
                "retriver": r"retriver_((?:(?!-model)[^_])+)",  # Fixed: allows hyphens within retriever name
                "model": r"model_(?:true_|false_)?([^-\s]+)",
                "attack": r"attack_([^-\s]+)",
                "removal": r"removal_([^-\s]+)",
                "adv_a_position": r"adv_a_position_([^-\s]+)",
                "score": r"-(dot|cos)",
                "adv_per_query": r"adv_per_query(\d+)",
                "M": r"-M(\d+)",  # Added M back based on folder name
                "repeat": r"Repeat(\d+)",  # Added Repeat back based on folder name
                "top_k": r"-top_(\d+)",  # Added M back based on folder name
                "no_questions": r"no_questions",
                "both_sides": r"both_sides",
                "num_tokens_removed": r"-num_tokens_removed_(\d+)",
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, experiment_name)
                if key in ["no_questions", "both_sides"]:
                    data[key] = 1 if match else 0
                else:
                    data[key] = match.group(1) if match else None
        else:
            # If experiment name is not found, set components to None
            for key in [
                "dataset",
                "retriver",  # Fixed: corrected spelling
                "model",
                "score",
                "attack",
                "adv_a_position",
                "removal",
                "adv_per_query",
                "M",
                "repeat",
                "top_k",
                "no_questions",
                "both_sides",
            ]:
                data[key] = None

        # Extract other desired patterns
        for key, pattern in desired_patterns.items():
            match = re.search(pattern, content)
            data[key] = match.group(1) if match else None

    return data


def check_log_completion(file_path):
    """
    Check if a log file has the required completion pattern at the end.
    Returns True if complete, False if incomplete.
    """
    required_patterns = [
        r"Success injection rate in top 5 contents: \d+\.\d+%",
        # r"Total run time for inference, removal and defense: \d+\.\d+",
        # r"Correct Answer Percentage: \d+\.\d+%",
        # r"Incorrect Answer Percentage: \d+\.\d+%",
        # r"Total run time for inference, removal and defense: \d+\.\d+",
    ]

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Check if all required patterns exist in the file
        for pattern in required_patterns:
            if not re.search(pattern, content):
                return False
        return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def find_incomplete_logs(logs_dir, delete_incomplete=False):
    """
    Find and optionally delete incomplete log files.
    Returns lists of complete and incomplete log files.
    """
    complete_logs = []
    incomplete_logs = []

    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)

                if check_log_completion(file_path):
                    complete_logs.append(file_path)
                else:
                    incomplete_logs.append(file_path)

    print("\n=== LOG COMPLETION ANALYSIS ===")
    print(f"Total log files found: {len(complete_logs) + len(incomplete_logs)}")
    print(f"Complete logs: {len(complete_logs)}")
    print(f"Incomplete logs: {len(incomplete_logs)}")

    if incomplete_logs:
        print("\n=== INCOMPLETE LOG FILES ===")
        for log_file in incomplete_logs:
            print(f"  - {log_file}")

        if delete_incomplete:
            print("\n=== DELETING INCOMPLETE LOGS ===")
            deleted_count = 0
            for log_file in incomplete_logs:
                try:
                    os.remove(log_file)
                    print(f"  Deleted: {log_file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {log_file}: {e}")
            print(f"Successfully deleted {deleted_count} incomplete log files.")
    else:
        print("All log files are complete!")

    return complete_logs, incomplete_logs


def clean_and_convert_data(df):
    """
    Cleans the DataFrame and converts percentage strings to floats,
    multiplying by 100 if the header contains 'rate' or 'pct'.
    """
    # Columns that represent percentages (and will be multiplied by 100)
    percentage_cols = [
        "success_injection_rate",
        "correct_answer_pct",
        "incorrect_answer_pct",
        "false_removal_rate",
        "success_removal_rate",
    ]

    for col in percentage_cols:
        if col in df.columns:  # Check if column exists in the DataFrame
            # Fill None/empty strings with NaN first, then apply conversion
            df[col] = df[col].replace("", np.nan).str.replace("%", "", regex=False)
            # Convert to float, coercing errors (e.g., if still non-numeric)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Multiply by 100
            if "rate" in col:
                df[col] = df[col] * 100.0

    # Convert inference_time to float
    if "inference_time" in df.columns:
        df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce")

    # Convert numeric columns that might have None
    for col in ["adv_per_query", "M", "repeat", "no_questions", "both_sides"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class ExperimentAnalyzer:
    """Clean, modular analyzer for experiment data with F1 score calculations."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "plots"):
        self.df = df.copy()
        self.output_dir = output_dir
        self.has_retriever = "retriver" in df.columns
        self.has_dataset = "dataset" in df.columns

        # Initialize retriever values
        if self.has_retriever:
            self.retriever_values = sorted(df["retriver"].dropna().unique())
            print(f"Found retriever types: {self.retriever_values}")
        else:
            self.retriever_values = [None]
            print(
                "Warning: 'retriever' column not found. Proceeding without retriever separation."
            )

        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")

    def calculate_f1_scores(self) -> pd.DataFrame:
        """Calculate F1, precision, and recall scores for the dataset."""

        def calculate_metrics(row):
            if pd.isna(row.get("success_removal_rate")) or pd.isna(
                row.get("false_removal_rate")
            ):
                return np.nan, np.nan, np.nan

            apq = row["adv_per_query"]
            m = row.get("M")
            repeat = row.get("repeat")
            top_k = row.get("top_k")

            num_origin = m * repeat * top_k
            num_adv = m * repeat * apq

            recall = row["success_removal_rate"] / 100
            specificity = row["false_removal_rate"] / 100
            tp = recall * num_adv
            fp = specificity * num_origin
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            if precision == 0 or recall == 0:
                return 0, precision, recall

            f1 = 2 * (precision * recall) / (precision + recall)
            return f1, precision, recall

        results = self.df.apply(calculate_metrics, axis=1)
        metrics_df = pd.DataFrame(
            results.tolist(),
            columns=["f1_score", "precision", "recall"],
            index=self.df.index,
        )

        self.df = pd.concat([self.df, metrics_df], axis=1)
        return self.df

    @staticmethod
    def create_display_label(row: pd.Series) -> str:
        """Create display labels for plots."""
        label_parts = []
        mappings = [
            ("dataset", "D"),
            ("model", "M"),
            ("adv_a_position", "P"),
            ("adv_per_query", "APQ"),
            ("top_k", "TOPK"),
        ]

        for col, prefix in mappings:
            if col in row.index and pd.notna(row[col]):
                label_parts.append(f"{prefix}:{row[col]}")

        return " ".join(label_parts)

    def get_plot_data(
        self,
        attack_filter=None,
        removal_filter=None,
        retriever_val=None,
    ) -> pd.DataFrame:
        """Get filtered and sorted data for plotting."""
        data = self.df.copy()

        # Apply filters
        if attack_filter:
            data = data[data["attack"] == attack_filter]
        if removal_filter:
            data = data[data["removal"].isin(removal_filter)]
        if self.has_retriever and retriever_val:
            data = data[data["retriver"] == retriever_val]

        # Sort data
        sort_keys = ["adv_per_query", "score", "dataset", "model", "adv_a_position"]
        if self.has_retriever:
            sort_keys.append("retriver")

        existing_keys = [key for key in sort_keys if key in data.columns]
        if existing_keys:
            data = data.sort_values(by=existing_keys)

        # Add display labels
        data["plot_specific_display_group"] = data.apply(
            self.create_display_label, axis=1
        )

        return data

    def create_plot(
        self,
        data: pd.DataFrame,
        metric: str,
        score_func: str,
        plot_type: str,
        retriever_val=None,
        comparison_mode: bool = False,
    ) -> str:
        """Create a single plot with consistent styling."""
        if data.empty or metric not in data.columns or data[metric].dropna().empty:
            return None

        plt.figure(figsize=(15, 7))

        # Determine colors and styling
        is_f1_score = metric == "f1_score"
        retriever_title = f" - Retriever: {retriever_val}" if retriever_val else ""
        title_metric = metric.replace("_", " ").title()

        if comparison_mode and "removal" in data.columns:
            # Comparison plot with hue
            palette = self._get_comparison_palette(
                data["removal"].unique(), is_f1_score
            )
            sns.barplot(
                x="plot_specific_display_group",
                y=metric,
                hue="removal",
                data=data,
                palette=palette,
            )
            plt.legend(title="Removal Method")
        else:
            # Single method plot
            color = self._get_single_color(plot_type, is_f1_score)
            sns.barplot(
                x="plot_specific_display_group", y=metric, data=data, color=color
            )

        # Set title and labels
        plt.title(
            f"{title_metric} ({plot_type}){retriever_title}\n"
            f"Score Func: {score_func} (AdvPerQuery sorted: 1s, 3s, 5s)"
        )
        plt.xlabel("Experiment Group (D:Dataset M:Model P:AdvPos APQ:AdvPerQuery)")
        plt.ylabel(title_metric)
        plt.xticks(rotation=75, ha="right", fontsize=8)

        # Add value annotations for F1 scores
        if is_f1_score:
            for container in plt.gca().containers:
                plt.gca().bar_label(container, fmt="%.3f", fontsize=8)

        plt.tight_layout()

        # Generate filename
        suffix = f"_retriever_{retriever_val}" if retriever_val else ""
        filename = os.path.join(
            self.output_dir, f"{plot_type}_{metric}_score_{score_func}{suffix}.png"
        )

        try:
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving plot {filename}: {e}")
            return None
        finally:
            plt.close()

    def _get_comparison_palette(self, removal_methods, is_f1_score: bool):
        """Get color palette for comparison plots."""
        base_colors = REMOVAL_COLORS

        # Extend with additional colors if needed
        all_colors = list(base_colors.values())
        if len(removal_methods) > len(base_colors):
            extra_colors = sns.color_palette(
                "Set2", n_colors=len(removal_methods) - len(base_colors)
            )
            all_colors.extend(extra_colors)

        return dict(zip(removal_methods, all_colors))

    def _get_single_color(self, plot_type: str, is_f1_score: bool) -> str:
        """Get single color for non-comparison plots."""
        color_map = {
            ("Hotflip Attack", True): "darkgray",
            ("Hotflip Attack", False): "gray",
            ("No Attack", True): "darkseagreen",
            ("No Attack", False): "mediumseagreen",
        }
        return color_map.get((plot_type, is_f1_score), "steelblue")

    def analyze_comparison(
        self,
        attack: str = "hotflip",
        removal_methods=None,
    ):
        """Analyze comparison between removal methods."""
        if removal_methods is None:
            removal_methods = REMOVAL_METHODS

        print(
            f"\n--- Analyzing {' vs '.join(removal_methods)} removal methods ({attack.title()} Attack) ---"
        )

        comparison_data = self.get_plot_data(
            attack_filter=attack, removal_filter=removal_methods
        )

        if comparison_data.empty:
            print(f"No data found for {attack} attack with {removal_methods} methods.")
            return

        # Find complete experimental pairs
        comparison_keys = [
            "dataset",
            "model",
            "score",
            "adv_a_position",
            "adv_per_query",
            "M",
            "repeat",
            "no_questions",
            "both_sides",
        ]
        if self.has_retriever:
            comparison_keys.append("retriver")

        existing_keys = [
            key for key in comparison_keys if key in comparison_data.columns
        ]
        comparison_data["experiment_setup_id"] = (
            comparison_data[existing_keys]
            .astype(str)
            .fillna("NA")
            .agg("_".join, axis=1)
        )

        setup_counts = comparison_data.groupby("experiment_setup_id")[
            "removal"
        ].nunique()
        complete_setups = setup_counts[setup_counts == len(removal_methods)].index
        comparable_data = comparison_data[
            comparison_data["experiment_setup_id"].isin(complete_setups)
        ]

        if comparable_data.empty:
            print("No complete pairs found for comparison.")
            return

        # Print summary
        metrics = [
            "success_injection_rate",
            "correct_answer_pct",
            "false_removal_rate",
            "success_removal_rate",
            "inference_time",
            "f1_score",
        ]
        existing_metrics = [m for m in metrics if m in comparable_data.columns]

        print(
            f"\n--- Comparison Summary: {' vs '.join(removal_methods)} ({attack.title()} Attack) ---"
        )
        # summary = (
        #     comparable_data.groupby(existing_keys + ["removal"])[existing_metrics]
        #     .mean()
        #     .unstack(level="removal")
        # )
        # print(summary.to_string())

        # Generate plots
        self._generate_plots_for_data(
            comparable_data,
            existing_metrics,
            "comparison",
            comparison_mode=True,
        )

    def analyze_single_method(self, attack: str, removal: str):
        """Analyze a single removal method under specific attack."""
        print(
            f"\n--- Analyzing '{removal}' removal method ({attack.title()} Attack) ---"
        )

        data = self.get_plot_data(attack_filter=attack, removal_filter=[removal])

        if data.empty:
            print(f"No data found for {attack} attack with {removal} method.")
            return

        metrics = [
            "success_injection_rate",
            "correct_answer_pct",
            "inference_time",
            "f1_score",
        ]
        existing_metrics = [m for m in metrics if m in data.columns]

        self._generate_plots_for_data(
            data, existing_metrics, f"{attack}_{removal}_removal", comparison_mode=False
        )

    def analyze_no_attack(self):
        """Analyze scenarios with no attack."""
        print("\n--- Analyzing 'none' attack scenarios (No Attack Baseline) ---")

        data = self.get_plot_data(attack_filter="none")

        if data.empty:
            print("No data found for 'none' attack scenarios.")
            return

        metrics = [
            "correct_answer_pct",
            "false_removal_rate",
            "inference_time",
            "success_injection_rate",
            "f1_score",
        ]
        existing_metrics = [m for m in metrics if m in data.columns]

        self._generate_plots_for_data(
            data,
            existing_metrics,
            "no_attack",
            comparison_mode=data["removal"].nunique() > 1,
        )

    def _generate_plots_for_data(
        self,
        data: pd.DataFrame,
        metrics,
        plot_type: str,
        comparison_mode: bool = False,
    ):
        """Generate plots for given data and metrics."""
        score_functions = ["dot", "cos"]

        for retriever_val in self.retriever_values:
            # Filter by retriever if applicable
            if self.has_retriever and retriever_val:
                retriever_data = data[data["retriver"] == retriever_val]
            else:
                retriever_data = data

            if retriever_data.empty:
                continue

            for metric in metrics:
                for score_func in score_functions:
                    plot_data = retriever_data[retriever_data["score"] == score_func]
                    if not plot_data.empty:
                        self.create_plot(
                            plot_data,
                            metric,
                            score_func,
                            plot_type,
                            retriever_val,
                            comparison_mode,
                        )

    def _create_combined_metrics_comparison_plot(
        self,
        avg_data: pd.DataFrame,
        metrics,
        attack: str,
        score_func: str,
        datasets,
        num_tokens,
    ):
        """Create a combined plot comparing removal methods across all metrics."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        colors = REMOVAL_COLORS

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(datasets))
            width = 0.22

            # Prepare scores for each removal method in REMOVAL_METHODS
            scores_dict = {method: [] for method in REMOVAL_METHODS}
            for dataset in datasets:
                dataset_data = avg_data[avg_data["dataset"] == dataset]
                for method in REMOVAL_METHODS:
                    row = dataset_data[dataset_data["removal"] == method]
                    score = row[metric].iloc[0] if not row.empty else np.nan
                    scores_dict[method].append(score)

            # Create bars for each method
            for i, method in enumerate(REMOVAL_METHODS):
                bars = ax.bar(
                    x + (i - len(REMOVAL_METHODS) // 2) * width,
                    scores_dict[method],
                    width,
                    label=method.title() if idx == 0 else "",
                    color=colors.get(method, "gray"),
                    alpha=0.8,
                )
                # Add value labels
                for bar, score in zip(bars, scores_dict[method]):
                    if not np.isnan(score):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            f"{score:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            # Customize subplot
            metric_title = metric.replace("_", " ").title()
            ax.set_title(metric_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Dataset", fontsize=10)
            ax.set_ylabel(f"Average {metric_title}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, 1.1)

        # Add overall title and legend
        attack_title = "No Attack" if attack == "none" else attack.title() + " Attack"
        score_title = "Cosine Similarity" if score_func == "cos" else "Dot Product"

        fig.suptitle(
            f"Performance Metrics Comparison: \n{attack_title} - {score_title}",
            fontsize=14,
            fontweight="bold",
        )

        # Add legend to the first subplot
        axes[0].legend()

        plt.tight_layout()

        # Save combined plot
        filename = f"dataset_combined_{attack}_attack_{score_func}_score_{num_tokens}_num_tokens.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved combined comparison plot: {filepath}")
        except Exception as e:
            print(f"Error saving combined comparison plot {filepath}: {e}")
        finally:
            plt.close()

    def analyze_f1_scores(self):
        """Comprehensive F1 score analysis."""
        if "f1_score" not in self.df.columns or self.df["f1_score"].dropna().empty:
            print("No F1 score data available for analysis.")
            return

        print("\n=== F1 SCORE DEEP DIVE ANALYSIS ===")

        # Group by key experimental factors
        groupby_cols = ["attack", "removal", "score"]
        if self.has_retriever:
            groupby_cols.append("retriver")
        if self.has_dataset:
            groupby_cols.insert(0, "dataset")

        f1_summary = (
            self.df.groupby(groupby_cols)["f1_score"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )

        print("\nF1 Score Summary:")
        print(f1_summary.to_string())

        if self.has_dataset:
            self._analyze_dataset_specific_f1()

    def _analyze_dataset_specific_f1(self):
        """Dataset-specific F1 score analysis."""
        print("\n=== DATASET-SPECIFIC F1 SCORE ANALYSIS ===")

        # Overall performance by dataset
        dataset_summary = (
            self.df.groupby("dataset")["f1_score"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        print("\nOverall F1 Score Performance by Dataset:")
        print(dataset_summary.to_string())

        # Best performing configurations per dataset
        print("\nBest F1 Score Configuration per Dataset:")
        best_configs = self.df.loc[self.df.groupby("dataset")["f1_score"].idxmax()]
        display_cols = ["dataset", "attack", "removal", "score", "f1_score"]
        if self.has_retriever:
            display_cols.insert(-1, "retriver")
        print(best_configs[display_cols].to_string(index=False))

        # Cross-dataset comparison
        print("\nF1 Score Comparison Across Datasets by Attack Type:")
        attack_dataset_comparison = self.df.pivot_table(
            values="f1_score", index="attack", columns="dataset", aggfunc="mean"
        ).round(4)
        print(attack_dataset_comparison.to_string())

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n--- Starting Complete Analysis ---")

        # Calculate F1 scores
        self.calculate_f1_scores()

        # Run different analysis sections
        # self.analyze_comparison(attack="hotflip", removal_methods=REMOVAL_METHODS)
        # self.analyze_single_method(attack="hotflip", removal="all")
        # self.analyze_no_attack()
        # self.analyze_f1_scores()

        # Create dataset performance barplots
        self.create_dataset_performance_barplots_by_tokens_removed()

        print("\n--- Finished Generating All Visualizations with F1 Score ---")

    def create_dataset_performance_barplots_by_tokens_removed(self):
        """
        Create barplots comparing removal methods for each dataset,
        separately for each value of num_tokens_removed.
        """
        print("\n--- Creating Dataset Performance Barplots by num_tokens_removed ---")

        if not self.has_dataset:
            print("No dataset column found. Skipping dataset performance barplots.")
            return

        if "num_tokens_removed" not in self.df.columns:
            print("No num_tokens_removed column found in data.")
            return

        filtered_data = self.df[self.df["removal"].isin(REMOVAL_METHODS)].copy()
        if getattr(self, "drift_apq5_only", False):
            # Only filter for drift removal
            mask = (filtered_data["removal"] == "drift") & (
                filtered_data["adv_per_query"] != 5
            )
            filtered_data = filtered_data[~mask]
        if filtered_data.empty:
            print(
                f"No data available for removal methods: {REMOVAL_METHODS}. Skipping dataset barplots."
            )
            return

        required_metrics = ["f1_score", "precision", "recall"]
        available_metrics = [m for m in required_metrics if m in filtered_data.columns]
        if not available_metrics:
            print(
                "No required metrics (f1_score, precision, recall) found. Skipping dataset barplots."
            )
            return

        attack_types = ["none", "hotflip", "aggd", "blackbox_hotflip"]
        score_functions = ["dot", "cos"]

        unique_tokens_removed = sorted(
            filtered_data["num_tokens_removed"].dropna().unique()
        )
        for num_tokens in unique_tokens_removed:
            data_tokens = filtered_data[
                filtered_data["num_tokens_removed"] == num_tokens
            ]
            if data_tokens.empty:
                continue

            for attack in attack_types:
                attack_data = data_tokens[data_tokens["attack"] == attack]
                if attack_data.empty:
                    print(
                        f"No data found for attack type: {attack} with num_tokens_removed={num_tokens}"
                    )
                    continue

                for score_func in score_functions:
                    score_data = attack_data[attack_data["score"] == score_func]
                    if score_data.empty:
                        print(
                            f"No data found for attack: {attack}, score: {score_func}, num_tokens_removed={num_tokens}"
                        )
                        continue

                    avg_data = (
                        score_data.groupby(["dataset", "removal"])[available_metrics]
                        .mean()
                        .reset_index()
                    )
                    datasets = avg_data["dataset"].unique()

                    for metric in available_metrics:
                        if avg_data[metric].dropna().empty:
                            print(
                                f"No valid data for metric: {metric}, attack: {attack}, score: {score_func}, num_tokens_removed={num_tokens}"
                            )
                            continue

                        plt.figure(figsize=(12, 7))
                        x = np.arange(len(datasets))
                        width = 0.22

                        scores_dict = {method: [] for method in REMOVAL_METHODS}
                        for dataset in datasets:
                            dataset_data = avg_data[avg_data["dataset"] == dataset]
                            for method in REMOVAL_METHODS:
                                row = dataset_data[dataset_data["removal"] == method]
                                score = row[metric].iloc[0] if not row.empty else np.nan
                                scores_dict[method].append(score)

                        colors = REMOVAL_COLORS

                        for i, method in enumerate(REMOVAL_METHODS):
                            bars = plt.bar(
                                x + (i - len(REMOVAL_METHODS) // 2) * width,
                                scores_dict[method],
                                width,
                                label=method.replace("_", " ").title(),
                                color=colors.get(method, "gray"),
                                alpha=0.8,
                            )
                            for bar, score in zip(bars, scores_dict[method]):
                                if not np.isnan(score):
                                    height = bar.get_height()
                                    plt.text(
                                        bar.get_x() + bar.get_width() / 2.0,
                                        height + 0.01,
                                        f"{score:.3f}",
                                        ha="center",
                                        va="bottom",
                                        fontweight="bold",
                                        fontsize=9,
                                    )

                        metric_title = metric.replace("_", " ").title()
                        attack_title = (
                            "No Attack"
                            if attack == "none"
                            else attack.title() + " Attack"
                        )
                        score_title = (
                            "Cosine Similarity"
                            if score_func == "cos"
                            else "Dot Product"
                        )

                        plt.title(
                            f"{metric_title} Comparison by Dataset: \n{attack_title} - {score_title}\nTokens Removed: {num_tokens}",
                            fontsize=14,
                            fontweight="bold",
                        )
                        plt.xlabel("Dataset", fontsize=12, fontweight="bold")
                        plt.ylabel(
                            f"Average {metric_title}", fontsize=12, fontweight="bold"
                        )
                        plt.xticks(x, datasets, rotation=45, ha="right")
                        plt.legend()

                        all_scores = [
                            s
                            for method_scores in scores_dict.values()
                            for s in method_scores
                            if not np.isnan(s)
                        ]
                        if all_scores:
                            y_min = max(0, min(all_scores) - 0.05)
                            y_max = min(1, max(all_scores) + 0.1)
                            plt.ylim(y_min, y_max)

                        plt.grid(axis="y", alpha=0.3)
                        plt.tight_layout()

                        filename = f"dataset_{metric}_{attack}_attack_{score_func}_tokens_removed_{num_tokens}.png"
                        filepath = os.path.join(self.output_dir, filename)

                        try:
                            plt.savefig(filepath, dpi=300, bbox_inches="tight")
                            print(f"Saved dataset comparison plot: {filepath}")
                        except Exception as e:
                            print(
                                f"Error saving dataset comparison plot {filepath}: {e}"
                            )
                        finally:
                            plt.close()

                    # Combined plot for all metrics
                    if len(available_metrics) > 1:
                        self._create_combined_metrics_comparison_plot(
                            avg_data,
                            available_metrics,
                            attack,
                            score_func,
                            datasets,
                            num_tokens,
                        )


def analyze_and_visualize(
    df: pd.DataFrame, output_dir: str = "plots", drift_apq5_only=False
):
    """
    Main function to analyze and visualize experiment data.

    Args:
        df: DataFrame containing experiment results
        output_dir: Directory to save plots
    """
    analyzer = ExperimentAnalyzer(df, output_dir)
    analyzer.drift_apq5_only = drift_apq5_only
    analyzer.run_complete_analysis()
    return analyzer.df


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze RAG experiment log data."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/home/tom.rahav/TrustRAG/logs",
        help="Path to the directory containing log files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/tom.rahav/TrustRAG/notebooks/extracted_logs_data.csv",
        help="Path to save the extracted CSV data.",
    )
    parser.add_argument(
        "--delete_incomplete",
        action="store_true",
        help="Delete incomplete log files (implies --check_incomplete).",
    )
    parser.add_argument(
        "--drift_apq5_only",
        action="store_true",
        help="For drift removal, only use data with adv_per_query == 5.",
    )
    args = parser.parse_args()

    # Check for incomplete logs if requested
    complete_logs, incomplete_logs = find_incomplete_logs(
        args.logs_dir, delete_incomplete=args.delete_incomplete
    )

    extracted_data = []
    for root, dirs, files in os.walk(args.logs_dir):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)

                # Only process complete log files
                if check_log_completion(file_path):
                    # print(f"Processing: {file_path}") # Suppress this print to reduce console clutter
                    data = parse_log_file(file_path)
                    if data:  # Only append if data was successfully extracted
                        extracted_data.append(data)
                else:
                    print(f"Skipping incomplete log: {file_path}")

    if not extracted_data:
        print("No log files found or no data extracted.")
        return

    # Dynamically get headers, ensuring 'experiment_full_name' is included
    # Also ensure consistent order for DictWriter
    all_keys = set()
    for row in extracted_data:
        all_keys.update(row.keys())
    headers = sorted(list(all_keys))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)

    print(f"Data extraction complete. Results saved to '{args.output_csv}'.")

    # --- Start Analysis ---
    print("\n--- Starting Data Analysis ---")
    df = pd.read_csv(args.output_csv)
    df = clean_and_convert_data(df)

    # Pass the plot directory to the analysis function
    analyze_and_visualize(df, output_dir="plots", drift_apq5_only=args.drift_apq5_only)


if __name__ == "__main__":
    main()
