import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from tqdm import tqdm

from clfextract.utils import (
    savefig,
    MODELS,
    MODELS_MAP,
    LAYER_MAP,
    MARKER_MAP,
    SOURCE_MAP,
)


sns.set_style("whitegrid")


def create_silhouette_lineplot(results_by_model, score_type, output_dir):
    """
    Create a lineplot showing silhouette scores across models and sources.
    score_type: either 'silhouette_y' or 'silhouette_y_pred'
    """
    model_source_data = defaultdict(lambda: defaultdict(list))
    unique_models = set()

    for model, results in results_by_model.items():
        if model not in MODELS:
            continue
        unique_models.add(model)
        for result in results:
            source = result["source"]
            layer = result["layer"]
            score = result[score_type]
            model_source_data[model][source].append((layer, score))

    colors = sns.color_palette("tab10", len(MODELS))
    color_map = {model: color for model, color in zip(MODELS, colors)}
    linestyles = ["solid", "dotted"]

    plt.figure()
    # To track added legend entries
    added_model_legend = set()
    added_source_legend = set()

    for source in SOURCE_MAP.keys():
        for model, sources in model_source_data.items():
            if source in sources:
                values = sources[source]
                values.sort(key=lambda x: x[0])  # Sort by layer
                layers, scores = zip(*values)
                fractions = [layer / LAYER_MAP[model] for layer in layers]

                # Plotting the line
                plt.plot(
                    fractions,
                    scores,
                    linestyle=linestyles[
                        list(SOURCE_MAP.keys()).index(source) % len(linestyles)
                    ],
                    color=color_map[model],
                    marker=MARKER_MAP[model],
                    label=(
                        MODELS_MAP[model] if model not in added_model_legend else None
                    ),  # Add model legend once
                    linewidth=1,
                    markersize=2,
                )
                # Add the model to the set
                added_model_legend.add(model)

                # Handle source legend (linestyle)
                if source not in added_source_legend:
                    added_source_legend.add(SOURCE_MAP[source])

    # Create custom legend entries for sources
    source_legend = [
        Line2D(
            [0],
            [0],
            linestyle=linestyles[
                list(SOURCE_MAP.keys()).index(source) % len(linestyles)
            ],
            color="black",
            label=SOURCE_MAP[source],
        )
        for source in SOURCE_MAP.keys()
    ]

    # Add legends for models (color) and sources (linestyles)
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color_map[model],
                label=MODELS_MAP[model],
                marker=MARKER_MAP[model],
            )
            for model in sorted(added_model_legend)
        ]
        + source_legend,
        loc="upper left",
    )

    plt.xlim(0, 1.05)
    plt.ylim(0, 0.6)
    plt.xlabel("Normalized Decoder Position")
    plt.ylabel("Silhouette Score")

    # savefig(os.path.join(output_dir, f"{score_type}_lineplot_{source}.pdf"))
    savefig(os.path.join(output_dir, f"{score_type}_lineplot.pdf"))


def create_component_heatmap_by_source(
    results, model, score_type, source, n_components, output_dir
):
    """
    Create a heatmap of silhouette scores for PCA components for a specific source.

    Args:
        results: List of results for a specific model
        model: Model name
        score_type: Either 'silhouette_y_pca' or 'silhouette_y_pred_pca'
        source: Specific source to plot
        n_components: Number of PCA components to show
        output_dir: Directory to save the plot
    """
    # Filter results for the specific source
    source_results = [r for r in results if r["source"] == source]

    if not source_results:
        return

    # Get all unique layers
    layers = sorted(list(set(r["layer"] for r in source_results)))

    # Create matrix for heatmap
    heatmap_data = np.zeros((n_components, len(layers)))

    # Fill the matrix
    for layer_idx, layer in enumerate(layers):
        layer_results = [r for r in source_results if r["layer"] == layer]
        if layer_results:
            avg_scores = np.mean(
                [r["pca"][score_type][:n_components] for r in layer_results], axis=0
            )
            heatmap_data[:, layer_idx] = avg_scores

    plt.figure()
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        xticklabels=range(0, len(layers), 5),
        yticklabels=range(n_components),
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Silhouette Score"},
    )
    plt.xticks(
        ticks=range(0, len(layers), 5), labels=range(0, len(layers), 5), rotation=0
    )
    plt.yticks(
        ticks=np.arange(0.5, n_components + 0.5),
        labels=range(1, n_components + 1),
        rotation=0,
    )
    plt.xlabel("Decoder")
    # Make ticks every 5 layers
    plt.ylabel("PCA Component")

    savefig(os.path.join(output_dir, f"{model}_{source}_{score_type}_heatmap.pdf"))


def process_results(results, output_dir, n_components=20):
    # Group results by model
    results_by_model = defaultdict(list)
    for result in results:
        model = result["model"]
        results_by_model[model].append(result)

    # Get unique sources
    sources = set(result["source"] for result in results)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "component_heatmaps"), exist_ok=True)

    # Create source-specific directories

    # Create general silhouette lineplots for each source
    for score_type in ["silhouette_y", "silhouette_y_pred"]:
        print(f"Creating silhouette lineplot for score type: {score_type}")
        create_silhouette_lineplot(results_by_model, score_type, output_dir)

    # # Create component heatmaps for each model and source
    # for model, model_results in tqdm(results_by_model.items()):
    #     if model not in MODELS:
    #         continue
    #     print(f"Creating component heatmaps for model: {model}")
    #     for source in sources:
    #         for score_type in ["silhouette_y_pca", "silhouette_y_pred_pca"]:
    #             create_component_heatmap_by_source(
    #                 model_results,
    #                 model,
    #                 score_type,
    #                 source,
    #                 n_components,
    #                 os.path.join(output_dir, "component_heatmaps"),
    #             )

    # print("Finished processing all results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for silhouette analysis results."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        default=["results.json"],
        help="Paths to the input JSON files containing all results (supports pattern matching)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="silhouette_visualizations",
        help="Base directory for output visualizations",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=5,
        help="Number of PCA components to analyze in heatmaps",
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        default=["llama2", "qwen2", "gemma1", "gemma2"],
        help="Models to include in the visualizations",
    )
    args = parser.parse_args()

    # Properly offset the model list for the color scheme
    for i in range(len(MODELS)):
        if MODELS[i] not in args.models:
            MODELS[i] = None

    all_results = []
    for pattern in args.input:
        for file_path in glob.glob(pattern):
            with open(file_path, "r") as f:
                all_results.extend(json.load(f))

    # Process all results
    process_results(all_results, args.output, args.n_components)

    print("All visualizations have been generated.")
