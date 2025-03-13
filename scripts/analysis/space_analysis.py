import argparse
import json
import os

import polars as pl
import torch
import torch.multiprocessing as mp
from clfextract.datasets import ParquetManager
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser(description="Create experiments")

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--attacks",
        type=str,
        required=True,
        help="Attacks to analyze (comma-separated)",
    )
    parser.add_argument(
        "--start_layer", type=int, required=True, help="Starting layer to analyze"
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        required=True,
        help="Ending layer to analyze (inclusive)",
    )
    parser.add_argument("--source", type=str, default="", help="Source dataset")
    parser.add_argument(
        "--n_components", type=int, default=2, help="Number of PCA components"
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run sequentially instead of parallel"
    )
    return parser.parse_args()


def load_all_data(
    mn,
    model,
    start_layer,
    end_layer,
    lens_type,
    attacks=["benign"],
    source="",
    data_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cache = {}
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR")

    # Load base filters
    base_filters = {}
    base_filters = {"layer": {"in": list(range(start_layer, end_layer + 1))}}
    base_filters["attack"] = {"in": attacks}  # TODO Fix
    columns = ["x", "num_positions", "layer", "y_true", "y_pred_protectai", "attack"]
    print("Loading all data from Parquet files...")
    # Load all relevant files
    data_files = os.listdir(data_dir)
    data_files = [f for f in data_files if f.endswith(".parquet")]
    files = [
        f
        for f in data_files
        if model in f and source in f and lens_type in f and "hard" not in f
    ]
    if len(files) == 0:
        raise ValueError("No files found to load.")

    print(f"Files found: {files}")

    # Load and concatenate data from all files
    embedding_df = pl.concat(
        [
            mn.load_dataset(
                os.path.join(data_dir, f), filters=base_filters, columns=columns
            )
            for f in files
        ]
    )

    embedding_df = embedding_df.rename({"y_pred_protectai": "y_pred"})

    # Process data for each layer
    for layer in range(start_layer, end_layer + 1):
        data_cache[layer] = {}

        layer_data = embedding_df.filter(pl.col("layer") == layer)
        x, y, y_pred = mn.load_torch_from_df(
            layer_data, aggregation="last", device=device
        )

        data_cache[layer]["embedding"] = {"x": x, "y": y, "y_pred": y_pred}

    return data_cache


def process_layer(args, layer, data_cache):

    layer_data = data_cache[layer]["embedding"]
    device = layer_data["x"].device

    # Standardize the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(layer_data["x"].cpu().numpy())

    silhouette_y = silhouette_score(
        x_scaled,
        layer_data["y"].cpu().numpy(),
        metric="cosine",
    )

    silhouette_y_pred = silhouette_score(
        x_scaled,
        layer_data["y_pred"].cpu().numpy(),
        metric="cosine",
    )

    results = {
        "layer": layer,
        "attacks": args.attacks.split("+"),
        "source": args.source,
        "model": args.model,
        "silhouette_y": float(silhouette_y),
        "silhouette_y_pred": float(silhouette_y_pred),
        "pca": {
            "n_components": args.n_components,
            "explained_variance_ratio": [],
            "silhouette_y_pca": [],
            "silhouette_y_pred_pca": [],
        },
    }

    # PCA
    pca = PCA(n_components=args.n_components)
    pca.fit(x_scaled)
    # Explained variance ratio
    results["pca"]["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    # Silhouette scores on each component
    for i in range(args.n_components):
        silhouette_y_pca = silhouette_score(
            pca.transform(x_scaled)[:, i].reshape(-1, 1),
            layer_data["y"].cpu().numpy(),
            metric="cosine",
        )
        silhouette_y_pred_pca = silhouette_score(
            pca.transform(x_scaled)[:, i].reshape(-1, 1),
            layer_data["y_pred"].cpu().numpy(),
            metric="cosine",
        )

        results["pca"]["silhouette_y_pca"].append(float(silhouette_y_pca))
        results["pca"]["silhouette_y_pred_pca"].append(float(silhouette_y_pred_pca))

    print(f"Layer results: {results}")

    return results


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = get_args()

    # Determine which types of data we need to load
    lens_type = "embedding"

    print("Loading data...")
    # Load all data once
    mn = ParquetManager()
    data_cache = load_all_data(
        mn,
        args.model,
        args.start_layer,
        args.end_layer,
        lens_type,
        attacks=["benign"],
        source=args.source,
        data_dir=args.input
    )

    print("Data loaded.")

    if args.sequential:
        combined_results = []
        for layer in range(args.start_layer, args.end_layer + 1):
            combined_results.append(process_layer(args, layer, data_cache))
    else:
        # Process layers in parallel
        with mp.Pool(processes=args.end_layer - args.start_layer + 1) as pool:
            combined_results = pool.starmap(
                process_layer,
                [
                    (args, layer, data_cache)
                    for layer in range(args.start_layer, args.end_layer + 1)
                ],
            )

    output_filename = (
        args.output
        + f"space_analysis_{args.source}_{args.model}_layer{args.start_layer:02d}-{args.end_layer:02d}_{args.attacks}.json"
    )

    with open(output_filename, "w") as f:
        json.dump(combined_results, f, indent=4)

    print(f"Results saved to {output_filename}")
