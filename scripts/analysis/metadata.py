import argparse
import json
import os

import polars as pl
import torch
from clfextract.datasets import ParquetManager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, default=os.getenv("DATA_DIR", ""), help="Input"
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory")

    return parser.parse_args()


def get_metadatas(
    mn,
    models,
    sources,
    attacks=["benign"],
    lens_type="embedding",
    data_dir=None,
):
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", ".")

    base_filters = {"layer": {"equals": 0}}
    columns = ["y_true", "y_pred_protectai", "y_pred_advbench", "attack"]

    # Load and concatenate data from all files
    metadatas = []

    for model in models:
        for source in sources:
            for attack in attacks:
                metadata = {
                    "model": model,
                    "source": source,
                    "attack": attack,
                    "lens_type": lens_type,
                    "y_pred_protectai": {},
                    "y_pred_advbench": {},
                }
                data_files = os.listdir(data_dir)
                data_files = [f for f in data_files if f.endswith(".parquet")]
                files = [
                    f
                    for f in data_files
                    if model in f and source in f and lens_type in f
                ]
                if len(files) == 0:
                    print(f"No files found for {model} and {source}")
                    continue

                filters = base_filters | {"attack": {"equals": attack}}

                embedding_df = pl.concat(
                    [
                        mn.load_dataset(
                            os.path.join(data_dir, f),
                            filters=filters,
                            columns=columns,
                        )
                        for f in files
                    ]
                )

                if embedding_df.shape[0] == 0:
                    print(f"No data found for {model}, {source}, and {attack}")
                    continue

                metadata["num_samples"] = int(embedding_df.shape[0])
                y_true, y_pred_protectai, y_pred_advbench = mn.load_torch_from_df(
                    embedding_df,
                    columns=["y_true", "y_pred_protectai", "y_pred_advbench"],
                )
                # Get the FP, FN, TP, TN
                tp = int(torch.sum(y_true & y_pred_protectai).item())
                tn = int(torch.sum(~y_true & ~y_pred_protectai).item())
                fp = int(torch.sum(~y_true & y_pred_protectai).item())
                fn = int(torch.sum(y_true & ~y_pred_protectai).item())
                f1 = float(2 * tp / (2 * tp + fp + fn))
                acc = float((tp + tn) / (tp + tn + fp + fn))

                metadata["y_pred_protectai"]["FP"] = fp
                metadata["y_pred_protectai"]["FN"] = fn
                metadata["y_pred_protectai"]["TP"] = tp
                metadata["y_pred_protectai"]["TN"] = tn
                metadata["y_pred_protectai"]["F1"] = f1
                metadata["y_pred_protectai"]["ACC"] = acc

                tp = int(torch.sum(y_true & y_pred_advbench).item())
                tn = int(torch.sum(~y_true & ~y_pred_advbench).item())
                fp = int(torch.sum(~y_true & y_pred_advbench).item())
                fn = int(torch.sum(y_true & ~y_pred_advbench).item())
                f1 = float(2 * tp / (2 * tp + fp + fn))
                acc = float((tp + tn) / (tp + tn + fp + fn))

                metadata["y_pred_advbench"]["FP"] = fp
                metadata["y_pred_advbench"]["FN"] = fn
                metadata["y_pred_advbench"]["TP"] = tp
                metadata["y_pred_advbench"]["TN"] = tn
                metadata["y_pred_advbench"]["F1"] = f1
                metadata["y_pred_advbench"]["ACC"] = acc

                metadata["agreement_y_pred"] = float(
                    torch.sum(y_pred_protectai == y_pred_advbench).item()
                    / len(y_pred_protectai)
                )

                metadatas.append(metadata)

    return metadatas


if __name__ == "__main__":
    print("Loading data...")
    args = get_args
    # Load all data once
    mn = ParquetManager()
    metadatas = get_metadatas(
        mn,
        [
            "gemma2",
            "gemma1",
            "granite",
            "llama2",
            "qwen2",
            "llama3",
            "mistral",
            "zephyrrmu",
        ],
        ["advbench", "or-bench"],
        ["benign", "gcg"],
        data_dir=args.input,
    )

    with open(args.output + "metadatas.json", "w") as f:
        json.dump(metadatas, f, indent=4)
