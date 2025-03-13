import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clfextract.utils import (
    savefig,
    MODELS,
    MODELS_MAP,
    REVERSE_MODELS_MAP,
    LAYER_MAP,
    DATASET_LINESTYLES,
    DATASET_MAP,
    MARKER_MAP,
)
from matplotlib.lines import Line2D


palette = sns.color_palette("tab10", len(MODELS_MAP.values()))
model_palette = {model: palette[i] for i, model in enumerate(MODELS_MAP.values())}
sns.set_style("whitegrid")


def get_args():
    parser = argparse.ArgumentParser(description="Visualize transfer learning results")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the JSON file containing the transfer results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Directory to save output plots and statistics",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Increase output verbosity",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["llama2", "qwen2", "gemma1", "gemma2"],
        help="Models to include in the visualization",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        default="",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--datasets",
        "-ds",
        nargs="+",
        default=["advbench_harmful.json", "or-bench_harmful.json"],
        help="Datasets to include in the visualization",
    )

    return parser.parse_args()


def load_json_data(json_file, verbose=False, subset=""):
    all_records = []

    with open(json_file, "r") as f:
        all_data = json.load(f)  # List of dictionaries

    for data in all_data:
        metadata = data["metadata"]
        if subset not in metadata["dataset"]:
            continue
        clf_layer = metadata["clf"]["layer"]
        clf_best_threshold = metadata["clf"]["best_threshold"]

        for record in data["records"]:
            for step, loss in enumerate(record.get("losses", [])):
                record_with_metadata = {
                    **record,
                    "best_threshold": clf_best_threshold,
                    "model": (
                        metadata["model"]
                        if "granite" not in metadata["model"]
                        else "granite"
                    ),
                    "dataset": metadata["dataset"],
                    "attack": metadata["attack"],
                    "layer": clf_layer,
                    "step": step,
                    "step_loss": loss,
                    "y_clf": record["y_clf"],
                    # success if the threshold is in between y_clf and y_clf_pre
                    "clf_success": (record["y_clf"] > clf_best_threshold)
                    != (record["y_clf_base"] > clf_best_threshold),
                    "train_acc": metadata["clf"]["train_acc"],
                    "test_acc": metadata["clf"]["test_acc"],
                    "train_f1": metadata["clf"]["train_f1"],
                    "test_f1": metadata["clf"]["test_f1"],
                }
                all_records.append(record_with_metadata)

    return pd.DataFrame(all_records)


def visualize_results(df, output_dir="results", verbose=False, dataset_name=""):
    os.makedirs(output_dir, exist_ok=True)

    # Only keep the ones that are successful for the clf
    # df = df[df['clf_success'] == True]

    if verbose:
        print(f"Saving plots to directory: {output_dir}")

    plt.figure()
    success_by_layer = (
        df.groupby(["layer", "model", "dataset"])["llm_success"].mean().reset_index()
    )

    # colors = sns.color_palette("tab10", len(MODELS))
    # color_map = {model: color for model, color in zip(MODELS, colors)}
    added_model_legend = set()
    added_source_legend = set()
    # Create the success rate plot with seaborn lineplot
    for dataset in success_by_layer["dataset"].unique():
        for model in success_by_layer["model"].unique():
            model_data = success_by_layer[success_by_layer["model"] == model]
            model_data = model_data[model_data["dataset"] == dataset]
            fraction = model_data["layer"] / LAYER_MAP[REVERSE_MODELS_MAP[model]]
            plt.plot(
                fraction,
                model_data["llm_success"],
                label=model,
                marker=MARKER_MAP[REVERSE_MODELS_MAP[model]],
                color=model_palette[model],
                linestyle=DATASET_LINESTYLES[dataset],
                linewidth=2,
                markersize=4,
            )

            added_model_legend.add(model)
            added_source_legend.add(dataset)

    dataset_legend = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DATASET_LINESTYLES[dataset],
            label=dataset,
        )
        for dataset in success_by_layer["dataset"].unique()
    ]

    # Add legends for models (color) and sources (linestyles)
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=model_palette[model],
                label=model,
                marker=MARKER_MAP[REVERSE_MODELS_MAP[model]],
            )
            for model in sorted(added_model_legend)
        ]
        + dataset_legend,
        loc="upper left",
    )
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel("Normalized Candidate Size")
    plt.ylabel("Transferability Rate")

    savefig(os.path.join(output_dir, f"transfer_clf_llm_{dataset_name}.pdf"))

    if verbose:
        print("Saved success rate by layer plot")


def compute_and_plot_asr(df, output_dir="results", verbose=False, dataset_name=""):
    """
    Compute and plot Attack Success Rate (ASR) for each layer and model

    Args:
        df (pd.DataFrame): Input dataframe with model results
        output_dir (str): Directory to save output plots
        verbose (bool): Enable verbose logging
    """
    # Group by model, layer, and take the last step's clf_success
    asr_by_layer = (
        df.groupby(["model", "layer", "dataset"])["clf_success"].mean().reset_index()
    )

    plt.figure()
    added_model_legend = set()
    added_source_legend = set()
    # Create the ASR plot with plt.plot
    for dataset in asr_by_layer["dataset"].unique():
        for model in asr_by_layer["model"].unique():
            model_data = asr_by_layer[
                (asr_by_layer["model"] == model) & (asr_by_layer["dataset"] == dataset)
            ]
            model_data = model_data[model_data["dataset"] == dataset]
            model_data["fraction"] = (
                model_data["layer"] / LAYER_MAP[REVERSE_MODELS_MAP[model]]
            )
            plt.plot(
                model_data["fraction"],
                model_data["clf_success"],
                label=model,
                marker=MARKER_MAP[REVERSE_MODELS_MAP[model]],
                color=model_palette[model],
                linestyle=DATASET_LINESTYLES[dataset],
                linewidth=2,
                markersize=4,
            )

            added_model_legend.add(model)
            added_source_legend.add(dataset)

    dataset_legend = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DATASET_LINESTYLES[dataset],
            label=dataset,
        )
        for dataset in asr_by_layer["dataset"].unique()
    ]

    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=model_palette[model],
                label=model,
                marker=MARKER_MAP[REVERSE_MODELS_MAP[model]],
            )
            for model in sorted(asr_by_layer["model"].unique())
        ]
        + dataset_legend,
        loc="lower left",
    )
    plt.xlabel("Normalized Candidate Size")
    plt.ylabel("Classifier ASR")
    plt.xlim(0, 1.05)  # Set x-axis from 0 to 1
    plt.ylim(0, 1.05)  # Set y-axis from 0 to 1

    savefig(os.path.join(output_dir, f"classifier_asr_{dataset_name}.pdf"))

    if verbose:
        print("Saved Classifier ASR by Layer plot")

    # Optional: Save ASR data to CSV for further analysis
    asr_by_layer.to_csv(
        os.path.join(output_dir, f"classifier_asr_{dataset_name}.csv"),
        index=False,
    )

    return asr_by_layer


if __name__ == "__main__":
    args = get_args()

    for i in range(len(MODELS)):
        if MODELS[i] not in args.models:
            MODELS[i] = None

    print(f"Loading data from {args.input}...")

    df = load_json_data(args.input, args.verbose, subset=args.dataset_name)

    print("Data loaded.")

    df = df[df["model"].isin(MODELS)]
    df["model"] = df["model"].map(MODELS_MAP)
    df["dataset"] = df["dataset"].map(DATASET_MAP)

    visualize_results(df, args.output, args.verbose, dataset_name=args.dataset_name)

    # Compute and plot ASR
    asr_by_layer = compute_and_plot_asr(
        df, args.output, args.verbose, dataset_name=args.dataset_name
    )

    layers = df["layer"].unique()
    layers.sort()
