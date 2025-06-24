import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np  # For NaN values


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
                "top_k": r"-Top_(\d+)",  # Added M back based on folder name
                "no_questions": r"no_questions",
                "both_sides": r"both_sides",
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
        r"Correct Answer Percentage: \d+\.\d+%",
        r"Incorrect Answer Percentage: \d+\.\d+%",
        r"Total run time for inference, removal and defense: \d+\.\d+",
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

    print(f"\n=== LOG COMPLETION ANALYSIS ===")
    print(f"Total log files found: {len(complete_logs) + len(incomplete_logs)}")
    print(f"Complete logs: {len(complete_logs)}")
    print(f"Incomplete logs: {len(incomplete_logs)}")

    if incomplete_logs:
        print(f"\n=== INCOMPLETE LOG FILES ===")
        for log_file in incomplete_logs:
            print(f"  - {log_file}")

        if delete_incomplete:
            print(f"\n=== DELETING INCOMPLETE LOGS ===")
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


def analyze_and_visualize(df, output_dir="plots"):
    """
    Complete analysis function with all original visualizations PLUS F1 score calculations.
    F1 Score is calculated from:
    - Precision = successful_removal_rate
    - Recall = 1 - false_removal_rate
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    print("\n--- Basic Statistics ---")
    # print(df.describe().to_string())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Check if retriever column exists
    has_retriever = "retriver" in df.columns
    if not has_retriever:
        print(
            "\nWarning: 'retriever' column not found. Proceeding without retriever separation."
        )
        retriever_values = [None]
    else:
        retriever_values = sorted(df["retriver"].dropna().unique())
        print(f"\nFound retriever types: {retriever_values}")

    # === F1 SCORE CALCULATION ===
    def calculate_f1_score(row):
        """Calculate F1 score from successful_removal_rate and false_removal_rate"""
        if pd.isna(row.get("success_removal_rate")) or pd.isna(
            row.get("false_removal_rate")
        ):
            return np.nan
        apq = row["adv_per_query"]
        m = row.get("M")
        repeat = row.get("repeat")
        top_k = row.get("top_k")

        num_origin = m * repeat * top_k
        num_adv = m * repeat * apq

        recall = (
            row["success_removal_rate"] / 100
        )  # 1 - FP / (TP + FP) = TP / (TP + FN)
        specificity = (
            row["false_removal_rate"] / 100
        )  # 1 - TN / (TN + FP) = FP / (TN + FP)
        tp = recall * num_adv
        fp = specificity * num_origin
        precision = tp / (tp + fp)  # TP / (TP + FP)

        if precision == 0 or recall == 0:
            return 0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    # Add F1 score column to dataframe
    df = df.copy()
    df["f1_score"] = df.apply(calculate_f1_score, axis=1)

    print("\n=== F1 SCORE STATISTICS ===")
    print(f"Mean F1 Score: {df['f1_score'].mean():.4f}")
    print(f"Median F1 Score: {df['f1_score'].median():.4f}")
    print(f"F1 Score Range: {df['f1_score'].min():.4f} - {df['f1_score'].max():.4f}")

    # Define helper function for x-axis labels globally for all plotting parts
    def create_display_label(row):
        label_parts = []
        if "dataset" in row.index and pd.notna(row["dataset"]):
            label_parts.append(f"D:{row['dataset']}")
        if "model" in row.index and pd.notna(row["model"]):
            label_parts.append(f"M:{row['model']}")
        if "adv_a_position" in row.index and pd.notna(row["adv_a_position"]):
            label_parts.append(f"P:{row['adv_a_position']}")
        if "adv_per_query" in row.index and pd.notna(row["adv_per_query"]):
            label_parts.append(f"APQ:{row['adv_per_query']}")
        if "top_k" in row.index and pd.notna(row["top_k"]):
            label_parts.append(f"TOPK:{row['top_k']}")
        return " ".join(label_parts)

    # Focus on 'hotflip' attack for comparison in Part 1 and Part 2
    df_hotflip = df[df["attack"] == "hotflip"].copy()

    if df_hotflip.empty:
        print(
            "\nNo 'hotflip' attack data found. Skipping Part 1 and Part 2 visualizations."
        )
    else:
        # --- Part 1: Comparison of 'drift' vs 'kmeans_ngram' under 'hotflip' attack ---
        print(
            "\n--- Analyzing 'drift' vs 'kmeans_ngram' removal methods (Hotflip Attack) ---"
        )

        comparison_keys = [
            "dataset",
            "retriver",
            "model",
            "score",
            "adv_a_position",
            "adv_per_query",
            "M",
            "repeat",
            "no_questions",
            "both_sides",
        ]
        if has_retriever:
            comparison_keys.append("retriever")

        df_filtered_removal_comparison = df_hotflip[
            df_hotflip["removal"].isin(["drift", "kmeans_ngram"])
        ].copy()

        if df_filtered_removal_comparison.empty:
            print(
                "\nNo 'drift' or 'kmeans_ngram' removal data for 'hotflip' attack found for comparison plots."
            )
        else:
            existing_comparison_keys_comp = [
                key
                for key in comparison_keys
                if key in df_filtered_removal_comparison.columns
            ]
            if not all(
                key in existing_comparison_keys_comp
                for key in ["score", "adv_per_query"]
            ):
                print(
                    "\nCritical error for comparison plots: 'score' or 'adv_per_query' columns are missing."
                )
            else:
                df_filtered_removal_comparison["experiment_setup_id"] = (
                    df_filtered_removal_comparison[existing_comparison_keys_comp]
                    .astype(str)
                    .fillna("NA")
                    .agg("_".join, axis=1)
                )
                setup_counts = df_filtered_removal_comparison.groupby(
                    "experiment_setup_id"
                )["removal"].nunique()
                complete_setups = setup_counts[setup_counts == 2].index
                df_comparable = df_filtered_removal_comparison[
                    df_filtered_removal_comparison["experiment_setup_id"].isin(
                        complete_setups
                    )
                ].copy()

                if df_comparable.empty:
                    print(
                        "\nNo complete pairs of 'drift' and 'kmeans_ngram' experiments found for comparison."
                    )
                else:
                    print(
                        "\n--- Comparison Summary: Drift vs. Kmeans_Ngram (Hotflip Attack) ---"
                    )
                    # === UPDATED TO INCLUDE F1 SCORE ===
                    comparison_metrics = [
                        "success_injection_rate",
                        "correct_answer_pct",
                        "false_removal_rate",
                        "success_removal_rate",
                        "inference_time",
                        "f1_score",  # ADDED F1 SCORE
                    ]
                    final_comparison_keys_comp = [
                        key for key in comparison_keys if key in df_comparable.columns
                    ]
                    comparison_summary = (
                        df_comparable.groupby(final_comparison_keys_comp + ["removal"])[
                            comparison_metrics
                        ]
                        .mean()
                        .dropna(how="all")
                        .unstack(level="removal")
                    )
                    print(comparison_summary.to_string())

                    print(
                        "\n--- Generating Visualizations for 'drift' vs 'kmeans_ngram' (Hotflip Attack) ---"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    sns.set_style("whitegrid")
                    if (
                        "score" not in df_comparable.columns
                        or "adv_per_query" not in df_comparable.columns
                    ):
                        print(
                            "Error: 'score' or 'adv_per_query' missing in df_comparable for comparison plots."
                        )
                    else:
                        score_function_values_to_iterate = ["dot", "cos"]
                        sort_keys_comp = [
                            "adv_per_query",
                            "score",
                            "dataset",
                            "model",
                            "adv_a_position",
                        ]
                        if has_retriever:
                            sort_keys_comp.append("retriever")

                        existing_sort_keys_comp = [
                            key
                            for key in sort_keys_comp
                            if key in df_comparable.columns
                        ]
                        if (
                            not existing_sort_keys_comp
                            or "adv_per_query" not in existing_sort_keys_comp
                        ):
                            print(
                                "Error: Necessary sort keys missing for df_comparable."
                            )
                        else:
                            df_comparable_sorted = df_comparable.sort_values(
                                by=existing_sort_keys_comp
                            )

                            # Iterate through retrievers
                            for retriever_val in retriever_values:
                                if has_retriever and retriever_val is not None:
                                    df_retriever_subset = df_comparable_sorted[
                                        df_comparable_sorted["retriver"]
                                        == retriever_val
                                    ].copy()
                                    retriever_suffix = f"_retriever_{retriever_val}"
                                    retriever_title = f" - Retriever: {retriever_val}"
                                else:
                                    df_retriever_subset = df_comparable_sorted.copy()
                                    retriever_suffix = ""
                                    retriever_title = ""

                                if df_retriever_subset.empty:
                                    continue

                                for metric in comparison_metrics:
                                    if (
                                        metric not in df_retriever_subset.columns
                                        or df_retriever_subset[metric].dropna().empty
                                    ):
                                        print(
                                            f"Skipping comparison plot for '{metric}'{retriever_title}: missing or all NaN."
                                        )
                                        continue
                                    for sf_value in score_function_values_to_iterate:
                                        df_plot_data_comp = df_retriever_subset[
                                            df_retriever_subset["score"] == sf_value
                                        ].copy()
                                        if df_plot_data_comp.empty:
                                            continue
                                        plt.figure(figsize=(15, 7))
                                        df_plot_data_comp[
                                            "plot_specific_display_group"
                                        ] = df_plot_data_comp.apply(
                                            create_display_label, axis=1
                                        )

                                        # === SPECIAL STYLING FOR F1 SCORE ===
                                        if metric == "f1_score":
                                            palette = {
                                                "drift": "darkblue",
                                                "kmeans_ngram": "darkred",
                                            }
                                        else:
                                            palette = {
                                                "drift": "skyblue",
                                                "kmeans_ngram": "salmon",
                                            }

                                        sns.barplot(
                                            x="plot_specific_display_group",
                                            y=metric,
                                            hue="removal",
                                            data=df_plot_data_comp,
                                            palette=palette,
                                        )
                                        title_metric_name = metric.replace(
                                            "_", " "
                                        ).title()
                                        plt.title(
                                            f"{title_metric_name} (Hotflip Attack) Comparison{retriever_title}\nScore Func: {sf_value} (AdvPerQuery sorted: 1s, 3s, 5s)"
                                        )
                                        plt.xlabel(
                                            "Experiment Group (D:Dataset M:Model P:AdvPos APQ:AdvPerQuery)"
                                        )
                                        plt.ylabel(title_metric_name)
                                        plt.xticks(rotation=75, ha="right", fontsize=8)
                                        plt.legend(title="Removal Method")

                                        # === ADD VALUE ANNOTATIONS FOR F1 SCORE ===
                                        if metric == "f1_score":
                                            for container in plt.gca().containers:
                                                plt.gca().bar_label(
                                                    container, fmt="%.3f", fontsize=8
                                                )

                                        plt.tight_layout()
                                        plot_filename = os.path.join(
                                            output_dir,
                                            f"comparison_{metric}_score_{sf_value}{retriever_suffix}.png",
                                        )
                                        try:
                                            plt.savefig(plot_filename)
                                            print(f"Saved plot to {plot_filename}")
                                        except Exception as e:
                                            print(
                                                f"Error saving plot {plot_filename}: {e}"
                                            )
                                        plt.close()

        # --- Part 2: Analysis of 'all' removal method (no context/baseline) under 'hotflip' attack ---
        print("\n--- Analyzing 'all' removal method (Hotflip Attack) ---")
        df_all_removal_hotflip = df[df["removal"] == "all"].copy()

        if df_all_removal_hotflip.empty:
            print(
                "\nNo data found for 'all' removal method under 'hotflip' attack. Skipping these plots."
            )
        else:
            if (
                "score" not in df_all_removal_hotflip.columns
                or "adv_per_query" not in df_all_removal_hotflip.columns
            ):
                print(
                    "Error: 'score' or 'adv_per_query' missing in df_all_removal_hotflip. Skipping 'all' removal plots for hotflip attack."
                )
            else:
                # === UPDATED TO INCLUDE F1 SCORE ===
                metrics_for_all_plots = [
                    "success_injection_rate",
                    "correct_answer_pct",
                    "inference_time",
                    "f1_score",  # ADDED F1 SCORE
                ]
                metrics_for_all_plots = [
                    m
                    for m in metrics_for_all_plots
                    if m in df_all_removal_hotflip.columns
                ]
                print(
                    "\n--- Generating Visualizations for 'all' removal method (Hotflip Attack) ---"
                )
                os.makedirs(output_dir, exist_ok=True)
                sns.set_style("whitegrid")
                score_function_values_to_iterate = ["dot", "cos"]
                sort_keys_all_hotflip = [
                    "adv_per_query",
                    "score",
                    "dataset",
                    "model",
                    "adv_a_position",
                ]
                if has_retriever:
                    sort_keys_all_hotflip.append("retriever")

                existing_sort_keys_all_hotflip = [
                    key
                    for key in sort_keys_all_hotflip
                    if key in df_all_removal_hotflip.columns
                ]
                if (
                    not existing_sort_keys_all_hotflip
                    or "adv_per_query" not in existing_sort_keys_all_hotflip
                ):
                    print(
                        "Error: Necessary sort keys missing for df_all_removal_hotflip."
                    )
                else:
                    df_all_removal_hotflip_sorted = df_all_removal_hotflip.sort_values(
                        by=existing_sort_keys_all_hotflip
                    )

                    # Iterate through retrievers
                    for retriever_val in retriever_values:
                        if has_retriever and retriever_val is not None:
                            df_retriever_subset = df_all_removal_hotflip_sorted[
                                df_all_removal_hotflip_sorted["retriver"]
                                == retriever_val
                            ].copy()
                            retriever_suffix = f"_retriever_{retriever_val}"
                            retriever_title = f" - Retriever: {retriever_val}"
                        else:
                            df_retriever_subset = df_all_removal_hotflip_sorted.copy()
                            retriever_suffix = ""
                            retriever_title = ""

                        if df_retriever_subset.empty:
                            continue

                        for metric in metrics_for_all_plots:
                            if (
                                metric not in df_retriever_subset.columns
                                or df_retriever_subset[metric].dropna().empty
                            ):
                                print(
                                    f"Skipping 'all' removal (hotflip) plot for '{metric}'{retriever_title}: missing or all NaN."
                                )
                                continue
                            for sf_value in score_function_values_to_iterate:
                                df_plot_data_all_hf = df_retriever_subset[
                                    df_retriever_subset["score"] == sf_value
                                ].copy()
                                if df_plot_data_all_hf.empty:
                                    continue
                                plt.figure(figsize=(15, 7))
                                df_plot_data_all_hf["plot_specific_display_group"] = (
                                    df_plot_data_all_hf.apply(
                                        create_display_label, axis=1
                                    )
                                )

                                # === SPECIAL COLOR FOR F1 SCORE ===
                                color = "darkgray" if metric == "f1_score" else "gray"

                                sns.barplot(
                                    x="plot_specific_display_group",
                                    y=metric,
                                    data=df_plot_data_all_hf,
                                    color=color,
                                )
                                title_metric_name = metric.replace("_", " ").title()
                                plt.title(
                                    f"{title_metric_name} (Hotflip Attack) - Removal: All{retriever_title}\nScore Func: {sf_value} (AdvPerQuery sorted: 1s, 3s, 5s)"
                                )
                                plt.xlabel(
                                    "Experiment Group (D:Dataset M:Model P:AdvPos APQ:AdvPerQuery)"
                                )
                                plt.ylabel(title_metric_name)
                                plt.xticks(rotation=75, ha="right", fontsize=8)

                                # === ADD VALUE ANNOTATIONS FOR F1 SCORE ===
                                if metric == "f1_score":
                                    for container in plt.gca().containers:
                                        plt.gca().bar_label(
                                            container, fmt="%.3f", fontsize=8
                                        )

                                plt.tight_layout()
                                plot_filename = os.path.join(
                                    output_dir,
                                    f"hotflip_all_removal_{metric}_score_{sf_value}{retriever_suffix}.png",
                                )
                                try:
                                    plt.savefig(plot_filename)
                                    print(f"Saved plot to {plot_filename}")
                                except Exception as e:
                                    print(f"Error saving plot {plot_filename}: {e}")
                                plt.close()

    # --- Part 3: Analysis of 'none' attack scenarios ---
    print("\n--- Analyzing 'none' attack scenarios (No Attack Baseline) ---")
    # Use the original df, not df_hotflip
    df_no_attack = df[df["attack"] == "none"].copy()

    if df_no_attack.empty:
        print("\nNo data found for 'none' attack scenarios. Skipping these plots.")
    else:
        if (
            "score" not in df_no_attack.columns
            or "adv_per_query" not in df_no_attack.columns
        ):
            print(
                "Error: 'score' or 'adv_per_query' missing in df_no_attack. Skipping 'none' attack plots."
            )
        else:
            # === UPDATED TO INCLUDE F1 SCORE ===
            metrics_for_no_attack_plots = [
                "correct_answer_pct",
                "false_removal_rate",
                "inference_time",
                "success_injection_rate",
                "f1_score",  # ADDED F1 SCORE
            ]
            metrics_for_no_attack_plots = [
                m for m in metrics_for_no_attack_plots if m in df_no_attack.columns
            ]

            print("\n--- Generating Visualizations for 'none' attack scenarios ---")
            os.makedirs(output_dir, exist_ok=True)
            sns.set_style("whitegrid")
            score_function_values_to_iterate = ["dot", "cos"]
            sort_keys_no_attack = [
                "adv_per_query",
                "score",
                "dataset",
                "model",
                "adv_a_position",
                "removal",
            ]
            if has_retriever:
                sort_keys_no_attack.append("retriever")

            existing_sort_keys_no_attack = [
                key for key in sort_keys_no_attack if key in df_no_attack.columns
            ]

            if (
                not existing_sort_keys_no_attack
                or "adv_per_query" not in existing_sort_keys_no_attack
            ):
                print(
                    "Error: Necessary sort keys missing for df_no_attack. Skipping 'none' attack plots."
                )
            else:
                df_no_attack_sorted = df_no_attack.sort_values(
                    by=existing_sort_keys_no_attack
                )

                # Iterate through retrievers
                for retriever_val in retriever_values:
                    if has_retriever and retriever_val is not None:
                        df_retriever_subset = df_no_attack_sorted[
                            df_no_attack_sorted["retriver"] == retriever_val
                        ].copy()
                        retriever_suffix = f"_retriever_{retriever_val}"
                        retriever_title = f" - Retriever: {retriever_val}"
                    else:
                        df_retriever_subset = df_no_attack_sorted.copy()
                        retriever_suffix = ""
                        retriever_title = ""

                    if df_retriever_subset.empty:
                        continue

                    for metric in metrics_for_no_attack_plots:
                        if (
                            metric not in df_retriever_subset.columns
                            or df_retriever_subset[metric].dropna().empty
                        ):
                            print(
                                f"Skipping 'none' attack plot for '{metric}'{retriever_title}: missing or all NaN."
                            )
                            continue
                        for sf_value in score_function_values_to_iterate:
                            df_plot_data_no_attack = df_retriever_subset[
                                df_retriever_subset["score"] == sf_value
                            ].copy()
                            if df_plot_data_no_attack.empty:
                                continue

                            plt.figure(figsize=(15, 7))
                            df_plot_data_no_attack["plot_specific_display_group"] = (
                                df_plot_data_no_attack.apply(
                                    create_display_label, axis=1
                                )
                            )

                            num_removal_methods = df_plot_data_no_attack[
                                "removal"
                            ].nunique()
                            title_metric_name = metric.replace("_", " ").title()

                            if num_removal_methods > 1:
                                # Dynamically create a palette if multiple removal methods are present
                                unique_removals = df_plot_data_no_attack[
                                    "removal"
                                ].unique()

                                # === SPECIAL PALETTE FOR F1 SCORE ===
                                if metric == "f1_score":
                                    palette_no_attack = sns.color_palette(
                                        "plasma", n_colors=len(unique_removals)
                                    )
                                else:
                                    palette_no_attack = sns.color_palette(
                                        "viridis", n_colors=len(unique_removals)
                                    )
                                removal_palette_map = dict(
                                    zip(unique_removals, palette_no_attack)
                                )

                                sns.barplot(
                                    x="plot_specific_display_group",
                                    y=metric,
                                    hue="removal",
                                    data=df_plot_data_no_attack,
                                    palette=removal_palette_map,
                                )
                                plt.legend(title="Removal Method")
                                plt.title(
                                    f"{title_metric_name} (No Attack){retriever_title}\nScore Func: {sf_value} (AdvPerQuery sorted: 1s, 3s, 5s)"
                                )
                            else:
                                # If only one removal method
                                single_removal_method_name = (
                                    df_plot_data_no_attack["removal"].iloc[0]
                                    if not df_plot_data_no_attack.empty
                                    else "Unknown"
                                )

                                # === SPECIAL COLOR FOR F1 SCORE ===
                                color = (
                                    "darkseagreen"
                                    if metric == "f1_score"
                                    else "mediumseagreen"
                                )

                                sns.barplot(
                                    x="plot_specific_display_group",
                                    y=metric,
                                    data=df_plot_data_no_attack,
                                    color=color,
                                )
                                plt.title(
                                    f"{title_metric_name} (No Attack) - Removal: {single_removal_method_name}{retriever_title}\nScore Func: {sf_value} (AdvPerQuery sorted: 1s, 3s, 5s)"
                                )

                            plt.xlabel(
                                "Experiment Group (D:Dataset M:Model P:AdvPos APQ:AdvPerQuery)"
                            )
                            plt.ylabel(title_metric_name)
                            plt.xticks(rotation=75, ha="right", fontsize=8)

                            # === ADD VALUE ANNOTATIONS FOR F1 SCORE ===
                            if metric == "f1_score":
                                for container in plt.gca().containers:
                                    plt.gca().bar_label(
                                        container, fmt="%.3f", fontsize=8
                                    )

                            plt.tight_layout()
                            plot_filename = os.path.join(
                                output_dir,
                                f"no_attack_{metric}_score_{sf_value}{retriever_suffix}.png",
                            )
                            try:
                                plt.savefig(plot_filename)
                                print(f"Saved plot to {plot_filename}")
                            except Exception as e:
                                print(f"Error saving plot {plot_filename}: {e}")
                            plt.close()

    # === ENHANCED F1 SCORE ANALYSIS WITH DATASET DIFFERENTIATION ===
    print("\n=== F1 SCORE DEEP DIVE ANALYSIS ===")

    if "f1_score" in df.columns and not df["f1_score"].dropna().empty:
        # Check if dataset column exists
        has_dataset = "dataset" in df.columns

        # Group by key experimental factors including dataset
        groupby_cols = ["attack", "removal", "score"]
        if has_retriever:
            groupby_cols.append("retriver")
        if has_dataset:
            groupby_cols.insert(0, "dataset")  # Add dataset as first grouping column

        f1_summary = (
            df.groupby(groupby_cols)["f1_score"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )

        if has_dataset:
            print(
                "\nF1 Score Summary by Dataset, Attack, Removal Method, and Score Function:"
            )
        else:
            print("\nF1 Score Summary by Attack, Removal Method, and Score Function:")
        print(f1_summary.to_string())

        # Dataset-specific analysis if dataset column exists
        if has_dataset:
            print("\n=== DATASET-SPECIFIC F1 SCORE ANALYSIS ===")

            # Overall performance by dataset
            dataset_summary = (
                df.groupby("dataset")["f1_score"]
                .agg(["count", "mean", "std", "min", "max"])
                .round(4)
            )
            print("\nOverall F1 Score Performance by Dataset:")
            print(dataset_summary.to_string())

            # Best performing configurations per dataset
            print("\nBest F1 Score Configuration per Dataset:")
            best_configs = df.loc[df.groupby("dataset")["f1_score"].idxmax()]
            display_cols = ["dataset", "attack", "removal", "score", "f1_score"]
            if has_retriever:
                display_cols.insert(-1, "retriver")
            print(best_configs[display_cols].to_string(index=False))

            # Cross-dataset comparison for each attack type
            print("\nF1 Score Comparison Across Datasets by Attack Type:")
            attack_dataset_comparison = df.pivot_table(
                values="f1_score", index="attack", columns="dataset", aggfunc="mean"
            ).round(4)
            print(attack_dataset_comparison.to_string())

        # Create visualizations
        if has_dataset:
            # Get unique score types
            score_types = df["score"].unique()
            print(f"Found score types: {score_types}")
            print(f"Data shape: {df.shape}")
            print(f"F1 score column info: {df['f1_score'].describe()}")

            # Create separate heatmaps for each score type
            for score_type in score_types:
                print(f"\nProcessing {score_type} score type...")
                score_data = df[df["score"] == score_type]
                print(f"Score data shape: {score_data.shape}")
                print(f"Score data F1 scores: {score_data['f1_score'].describe()}")
                print(
                    f"Unique datasets: {score_data['dataset'].unique() if 'dataset' in score_data.columns else 'No dataset column'}"
                )
                print(
                    f"Unique attacks: {score_data['attack'].unique() if 'attack' in score_data.columns else 'No attack column'}"
                )
                print(
                    f"Unique removals: {score_data['removal'].unique() if 'removal' in score_data.columns else 'No removal column'}"
                )

                # Skip if no data for this score type
                if score_data.empty or score_data["f1_score"].dropna().empty:
                    print(
                        f"No F1 score data available for {score_type} score type, skipping heatmaps"
                    )
                    continue

                # Check required columns exist
                required_cols = ["dataset", "attack", "removal", "f1_score"]
                missing_cols = [
                    col for col in required_cols if col not in score_data.columns
                ]
                if missing_cols:
                    print(
                        f"Missing required columns for {score_type}: {missing_cols}, skipping"
                    )
                    continue

                print(
                    f"Successfully validated data for {score_type} - would create visualizations here"
                )
                # Heatmaps temporarily disabled for debugging

            # Create separate box plots for each score type
            for score_type in score_types:
                score_data = df[df["score"] == score_type]

                # Skip if no data for this score type
                if score_data.empty or score_data["f1_score"].dropna().empty:
                    print(
                        f"No F1 score data available for {score_type} score type, skipping boxplots"
                    )
                    continue

                print(
                    f"Would create boxplots for {score_type} score type ({len(score_data)} data points)"
                )
                # Boxplots temporarily disabled for debugging

        else:
            # Original single heatmap if no dataset column
            plt.figure(figsize=(12, 8))

            pivot_data = df.pivot_table(
                values="f1_score",
                index=["attack", "removal"],
                columns="score",
                aggfunc="mean",
            )

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                cbar_kws={"label": "F1 Score"},
            )
            plt.title(
                "F1 Score Heatmap by Attack Type, Removal Method, and Score Function"
            )
            plt.tight_layout()

            heatmap_filename = os.path.join(output_dir, "f1_score_heatmap.png")
            try:
                plt.savefig(heatmap_filename)
                print(f"Saved F1 heatmap to {heatmap_filename}")
            except Exception as e:
                print(f"Error saving F1 heatmap {heatmap_filename}: {e}")
            plt.close()

    else:
        print("No F1 score data available for analysis.")

    print("\n--- Finished Generating All Visualizations with F1 Score ---")


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
    analyze_and_visualize(df)


if __name__ == "__main__":
    main()
