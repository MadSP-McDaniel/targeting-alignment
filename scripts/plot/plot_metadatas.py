import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from clfextract.utils import savefig, MODELS, MODELS_MAP

sns.set_style("whitegrid")


def csv_to_latex_table(csv_file):
    """
    Convert consolidated CSV to LaTeX table

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        str: LaTeX tabular code
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Prepare LaTeX table
    latex_table = "\\begin{tabular}{l|cccc}\n\\hline\n"
    latex_table += "\\multirow{2}{*}{Model} & \\multicolumn{2}{c}{AdvBench} & \\multicolumn{2}{c}{OR-Bench} \\\\\n"
    # Dynamically determine the columns based on the CSV headers
    columns = df.columns[1:]  # Exclude the 'Model' column
    latex_table += "& " + " & ".join(columns) + " \\\\\n\\hline\n"

    # Add rows to LaTeX table
    for _, row in df.iterrows():
        latex_table += (
            f"{row['Model']} & "
            + " & ".join(str(row[col]) for col in columns)
            + " \\\\\n"
        )

    latex_table += "\\hline\n\\end{tabular}"

    return latex_table


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to JSON file containing performance data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output directory to save CSVs",
    )
    return parser.parse_args()


# Function to plot heatmap
def plot_heatmap(matrix, title, output_dir):
    plt.figure()
    # Normalize color scale
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        annot_kws={"size": 18},
        xticklabels=["Unsafe", "Safe"],
        yticklabels=["Unsafe", "Safe"],
        vmin=0,
        vmax=max(matrix[0][0] + matrix[1][0], matrix[0][1] + matrix[1][1]),
        cbar="qwen" in title.lower() or "zephyr" in title.lower(),
    )
    plt.ylabel("Predicted", fontsize=16)
    plt.xlabel("Actual", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if "gcg" in title.lower():
        savefig(
            os.path.join(output_dir, "gcg", f"{title.replace(' ', '_').lower()}.pdf")
        )
    else:
        savefig(
            os.path.join(output_dir, "benign", f"{title.replace(' ', '_').lower()}.pdf")
        )


if __name__ == "__main__":
    # Get arguments
    args = get_args()

    # Load JSON data
    with open(args.input, "r") as f:
        json_data = json.load(f)

    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare data for consolidated CSV
    model_performance = {}

    # Collect performance data
    for record in json_data:
        if record["model"] not in MODELS:
            continue
        # # Special condition for GCG attack: only use AdvBench
        # if record["attack"] == "gcg" and record["source"] == "or-bench":
        #     continue

        model = MODELS_MAP.get(record["model"], record["model"])
        source = record["source"]

        if model not in model_performance:
            model_performance[model] = {}

        # Store performance metrics
        metrics = record["y_pred_protectai"]

        # Calculate Attack Success Rate (ASR) as complement of False Negative Rate
        if source == "advbench":
            fnr = (
                metrics["FN"] / (metrics["FN"] + metrics["TP"])
                if (metrics["FN"] + metrics["TP"]) > 0
                else 0
            )
            asr = 1 - fnr

            model_performance[model].update(
                {
                    "Accuracy AdvBench": round(metrics["ACC"], 2),
                    "F1 AdvBench": round(metrics["F1"], 2),
                    "ASR AdvBench": round(asr, 2),
                }
            )
        elif source == "or-bench":
            model_performance[model].update(
                {
                    "Accuracy OR-Bench": round(metrics["ACC"], 2),
                    "F1 OR-Bench": round(metrics["F1"], 2),
                }
            )

    # Prepare CSV and LaTeX for each attack
    for attack in set(record["attack"] for record in json_data):
        attack_performance = {model: {} for model in MODELS_MAP.values()}

        # Collect performance data for each attack
        for record in json_data:
            if record["model"] not in MODELS or record["attack"] != attack:
                continue

            model = MODELS_MAP.get(record["model"], record["model"])
            source = record["source"]
            metrics = record["y_pred_protectai"]

            # Calculate Attack Success Rate (ASR)
            if source == "advbench":
                fnr = (
                    metrics["FN"] / (metrics["FN"] + metrics["TP"])
                    if (metrics["FN"] + metrics["TP"]) > 0
                    else 0
                )
                fpr = (
                    metrics["FP"] / (metrics["FP"] + metrics["TN"])
                    if (metrics["FP"] + metrics["TN"]) > 0
                    else 0
                )

                attack_performance[model].update(
                    {
                        "Accuracy AdvBench": round(metrics["ACC"], 2),
                        "F1 AdvBench": round(metrics["F1"], 2),
                        "ASR Unsafe AdvBench": round(fnr, 2),
                        "ASR Safe AdvBench": round(fpr, 2),
                    }
                )
            elif source == "or-bench":
                fnr = (
                    metrics["FN"] / (metrics["FN"] + metrics["TP"])
                    if (metrics["FN"] + metrics["TP"]) > 0
                    else 0
                )
                fpr = (
                    metrics["FP"] / (metrics["FP"] + metrics["TN"])
                    if (metrics["FP"] + metrics["TN"]) > 0
                    else 0
                )

                attack_performance[model].update(
                    {
                        "Accuracy OR-Bench": round(metrics["ACC"], 2),
                        "F1 OR-Bench": round(metrics["F1"], 2),
                        "ASR Unsafe OR-Bench": round(fnr, 2),
                        "ASR Safe OR-Bench": round(fpr, 2),
                    }
                )

        # Prepare CSV filename
        csv_filename = f"model_performance_{attack}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Write consolidated CSV
        with open(csv_path, "w", newline="") as csvfile:
            # Dynamically determine fieldnames based on attack
            fieldnames = ["Model"]
            if attack == "gcg":
                fieldnames.extend(
                    [
                        # "Accuracy AdvBench",
                        # "F1 AdvBench",
                        "ASR Unsafe AdvBench",
                        "ASR Safe AdvBench",
                        # "Accuracy OR-Bench",
                        # "F1 OR-Bench",
                        "ASR Unsafe OR-Bench",
                        "ASR Safe OR-Bench",
                    ]
                )
            else:
                fieldnames.extend(
                    [
                        "Accuracy AdvBench",
                        "F1 AdvBench",
                        # 'ASR Unsafe AdvBench',
                        # 'ASR Safe AdvBench',
                        "Accuracy OR-Bench",
                        "F1 OR-Bench",
                        # 'ASR Unsafe OR-Bench',
                        # 'ASR Safe OR-Bench'
                    ]
                )

            csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csvwriter.writeheader()

            # Write rows
            for model, performance in attack_performance.items():
                row = {"Model": model}
                performance = {k: v for k, v in performance.items() if k in fieldnames}
                row.update(performance)
                csvwriter.writerow(row)

        # Convert CSV to LaTeX table
        latex_table = csv_to_latex_table(csv_path)

        # Save LaTeX table to file
        latex_filename = f"model_performance_{attack}.tex"
        with open(os.path.join(output_dir, latex_filename), "w") as f:
            f.write(latex_table)

        print(f"CSV saved to: {csv_path}")
        print(f"LaTeX table saved to: {os.path.join(output_dir, latex_filename)}")

    # Prepare data for confusion matrices
    confusion_matrices = {}

    for record in json_data:
        if record["model"] not in MODELS:
            continue

        model = MODELS_MAP.get(record["model"], record["model"])
        source = record["source"]
        attack = record["attack"]

        if model not in confusion_matrices:
            confusion_matrices[model] = {}

        if attack not in confusion_matrices[model]:
            confusion_matrices[model][attack] = {}

        # Store confusion matrix based on source
        if source == "advbench":
            confusion_matrices[model][attack].update(
                {
                    "TP AdvBench": record["y_pred_protectai"]["TP"],
                    "FP AdvBench": record["y_pred_protectai"]["FP"],
                    "FN AdvBench": record["y_pred_protectai"]["FN"],
                    "TN AdvBench": record["y_pred_protectai"]["TN"],
                }
            )
        elif source == "or-bench":
            confusion_matrices[model][attack].update(
                {
                    "TP OR-Bench": record["y_pred_protectai"]["TP"],
                    "FP OR-Bench": record["y_pred_protectai"]["FP"],
                    "FN OR-Bench": record["y_pred_protectai"]["FN"],
                    "TN OR-Bench": record["y_pred_protectai"]["TN"],
                }
            )

    # Plot heatmaps and write CSVs for each attack
    for model, attacks in confusion_matrices.items():
        print(f"Creating confusion matrices for {model}")
        for attack, confusion in attacks.items():
            if attack == "gcg":
                advbench_matrix = [
                    [confusion.get("TP AdvBench", 0), confusion.get("FP AdvBench", 0)],
                    [confusion.get("FN AdvBench", 0), confusion.get("TN AdvBench", 0)],
                ]
                orbench_matrix = [
                    [confusion.get("TP OR-Bench", 0), confusion.get("FP OR-Bench", 0)],
                    [confusion.get("FN OR-Bench", 0), confusion.get("TN OR-Bench", 0)],
                ]

                plot_heatmap(
                    advbench_matrix,
                    f"{model} {attack} AdvBench Confusion Matrix",
                    output_dir,
                )

                plot_heatmap(
                    orbench_matrix,
                    f"{model} {attack} OR-Bench Confusion Matrix",
                    output_dir,
                )
            else:
                # Plot both AdvBench and OR-Bench matrices
                advbench_matrix = [
                    [confusion.get("TP AdvBench", 0), confusion.get("FP AdvBench", 0)],
                    [confusion.get("FN AdvBench", 0), confusion.get("TN AdvBench", 0)],
                ]
                orbench_matrix = [
                    [confusion.get("TP OR-Bench", 0), confusion.get("FP OR-Bench", 0)],
                    [confusion.get("FN OR-Bench", 0), confusion.get("TN OR-Bench", 0)],
                ]

                plot_heatmap(
                    advbench_matrix,
                    f"{model} {attack} AdvBench Confusion Matrix",
                    output_dir,
                )
                plot_heatmap(
                    orbench_matrix,
                    f"{model} {attack} OR-Bench Confusion Matrix",
                    output_dir,
                )
