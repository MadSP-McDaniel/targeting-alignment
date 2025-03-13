import argparse
import json
import os

import numpy as np
import polars as pl
import torch
import torch.multiprocessing as mp
from clfextract.classifiers import (
    LinearClassifier,
    RNNClassifier,
    MLPClassifier
)
from clfextract.datasets import ParquetManager
from sklearn.model_selection import StratifiedKFold


def get_args():
    parser = argparse.ArgumentParser(description="Create experiments")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=os.getenv("DATA_DIR", "./"),
        help="Input directory",
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
    parser.add_argument(
        "--train",
        type=str,
        default="y_true",
        choices=["y_true", "y_pred"],
        help="Labels to train on",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="Linear",
        choices=["Linear", "RNN", "MLP"],
        help="Type of classifier to use",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for RNN classifier"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers for RNN classifier"
    )
    parser.add_argument(
        "--analyze_heads",
        action="store_true",
        help="Whether to analyze individual heads",
    )
    parser.add_argument(
        "--early_stopping", action="store_true", help="Whether to use early stopping"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--n_splits", type=int, default=1, help="Number of splits for cross-validation"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--source", type=str, default="", help="Source dataset")
    parser.add_argument(
        "--sequential", action="store_true", help="Whether to run sequentially"
    )
    parser.add_argument(
        "--cross_source", type=str, default="", help="Cross-source dataset"
    )
    return parser.parse_args()


def train_and_evaluate(
    x,
    y,
    y_pred,
    bases,
    classifier_class,
    classifier_args,
    n_splits=1,
    early_stopping=False,
    patience=5,
    num_epochs=100,
    train_on="y_pred",
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    results = {
        "train": {
            "true_acc": [],
            "true_f1": [],
            "agreement_llm": [],
            "agreement_llm_f1": [],
        },
        "test": {
            "true_acc": [],
            "true_f1": [],
            "agreement_llm": [],
            "agreement_llm_f1": [],
        },
        "metadatas": [],
    }

    classifiers = []

    # Convert bases to numpy for splitting if it's a tensor

    split = (
        skf.split(x.cpu(), y_pred.cpu())
        if train_on == "y_pred"
        else skf.split(x.cpu(), y.cpu())
    )

    metadatas = []
    split_bases = []
    for train_index, test_index in split:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred_train, y_pred_test = y_pred[train_index], y_pred[test_index]
        bases_train, bases_test = bases[train_index], bases[test_index]

        # Store the base indices for this split
        split_bases.append({"train": bases_train.tolist(), "test": bases_test.tolist()})

        # Get the data TPR, TNR, FPR, FNR
        tpr = ((y_pred_train == y_train) & (y_train == 1)).sum().item() / (
            y_train == 1
        ).sum().item()
        tnr = ((y_pred_train == y_train) & (y_train == 0)).sum().item() / (
            y_train == 0
        ).sum().item()
        fpr = ((y_pred_train != y_train) & (y_train == 0)).sum().item() / (
            y_train == 0
        ).sum().item()
        fnr = ((y_pred_train != y_train) & (y_train == 1)).sum().item() / (
            y_train == 1
        ).sum().item()

        num_train, num_test = len(y_train), len(y_test)

        clf = classifier_class(**classifier_args)

        if train_on == "y_pred":
            losses = clf.train(
                x_train,
                y_pred_train,
                num_epochs=num_epochs,
                patience=patience if early_stopping else None,
            )
            pr_precision, pr_recall, pr_thresholds = clf.precision_recall_curve(
                x_train, y_pred_train
            )
            f_score = 2 * (pr_precision * pr_recall) / (pr_precision + pr_recall)
            best_threshold = pr_thresholds[np.argmax(f_score)]

        else:
            losses = clf.train(
                x_train,
                y_train,
                num_epochs=num_epochs,
                patience=patience if early_stopping else None,
            )
            pr_precision, pr_recall, pr_thresholds = clf.precision_recall_curve(
                x_train, y_train
            )
            f_score = 2 * (pr_precision * pr_recall) / (pr_precision + pr_recall)
            best_threshold = pr_thresholds[np.argmax(f_score)]

        metadatas.append(
            {
                "fold": len(metadatas),
                "best_threshold": float(best_threshold),
                "tpr": tpr,
                "tnr": tnr,
                "fpr": fpr,
                "fnr": fnr,
                "num_train": num_train,
                "num_test": num_test,
                "num_epochs": len(losses),
                "final_loss": losses[-1],
                "losses": losses,
            }
        )

        train_eval_true = clf.evaluate(x_train, y_train, threshold=best_threshold)
        train_eval_pred = clf.evaluate(x_train, y_pred_train, threshold=best_threshold)
        test_eval_true = clf.evaluate(x_test, y_test, threshold=best_threshold)
        test_eval_pred = clf.evaluate(x_test, y_pred_test, threshold=best_threshold)

        fold_results = {
            "train": {
                "true_acc": train_eval_true[0],
                "true_f1": train_eval_true[1],
                "agreement_llm": train_eval_pred[0],
                "agreement_llm_f1": train_eval_pred[1],
            },
            "test": {
                "true_acc": test_eval_true[0],
                "true_f1": test_eval_true[1],
                "agreement_llm": test_eval_pred[0],
                "agreement_llm_f1": test_eval_pred[1],
                "best_threshold": float(best_threshold),
            },
        }

        for split in ["train", "test"]:
            for metric in [
                "true_acc",
                "agreement_llm",
                "true_f1",
                "agreement_llm_f1",
            ]:
                results[split][metric].append(fold_results[split][metric])

        classifiers.append(clf)

    results["metadatas"] = metadatas

    return results, classifiers, split_bases


def load_all_data(
    mn,
    model,
    start_layer,
    end_layer,
    attacks=["benign"],
    source="",
    cross_source="",
    data_dir="",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cache = {}

    base_filters = {"layer": {"in": list(range(start_layer, end_layer + 1))}}
    columns = [
        "x",
        "num_positions",
        "layer",
        "y_true",
        "y_pred_protectai",
        "attack",
        "base",
    ]
    print("Loading all data from Parquet files...")
    data_files = os.listdir(data_dir)
    data_files = [f for f in data_files if f.endswith(".parquet")]
    files = [
        f
        for f in data_files
        if model in f and source in f
    ]
    if len(files) == 0:
        raise ValueError("No files found to load.")

    print(f"Files found: {files}")

    embedding_df = pl.concat(
        [
            mn.load_dataset(
                os.path.join(data_dir, f), filters=base_filters, columns=columns
            )
            for f in files
        ]
    )

    embedding_df = embedding_df.rename({"y_pred_protectai": "y_pred"})

    for attack in attacks:
        attack_data_cache = {}
        attack_df = embedding_df.filter(pl.col("attack") == attack)
        if len(attack_df) == 0:
            print(
                f"No data found for attack {attack} in cross-source data. Skipping..."
            )
            continue
        for layer in range(start_layer, end_layer + 1):
            attack_data_cache[layer] = {}

            layer_data = attack_df.filter(pl.col("layer") == layer)
            x, y, y_pred = mn.load_torch_from_df(
                layer_data,
                aggregation="last",
                device=device,
            )
            bases = layer_data["base"].to_numpy()

            attack_data_cache[layer]["embedding"] = {
                "x": x,
                "y": y,
                "y_pred": y_pred,
                "bases": bases,
            }

        data_cache[attack] = attack_data_cache

    if cross_source:
        cross_source_files = [
            f
            for f in data_files
            if model in f
            and cross_source in f
            and "embedding" in f
            and not f.startswith("gcg")
        ]
        if len(cross_source_files) == 0:
            raise ValueError("No cross-source files found to load.")

        print(f"Cross-source files found: {cross_source_files}")

        cross_source_df = pl.concat(
            [
                mn.load_dataset(
                    os.path.join(data_dir, f), filters=base_filters, columns=columns
                )
                for f in cross_source_files
            ]
        )

        cross_source_df = cross_source_df.rename({"y_pred_protectai": "y_pred"})

        data_cache["cross_source"] = {}
        for attack in attacks:
            attack_data_cache = {}
            attack_df = cross_source_df.filter(pl.col("attack") == attack)
            if len(attack_df) == 0:
                print(
                    f"No data found for attack {attack} in cross-source data. Skipping..."
                )
                continue
            for layer in range(start_layer, end_layer + 1):
                attack_data_cache[layer] = {}

                layer_data = attack_df.filter(pl.col("layer") == layer)
                x, y, y_pred = mn.load_torch_from_df(
                    layer_data,
                    aggregation="last",
                    device=device,
                )
                bases = layer_data["base"].to_numpy()

                attack_data_cache[layer]["embedding"] = {
                    "x": x,
                    "y": y,
                    "y_pred": y_pred,
                    "bases": bases,
                }

            data_cache["cross_source"][attack] = attack_data_cache

    return data_cache


def process_layer(args, layer, data_cache):
    classifier_args = {"input_dim": None, "learning_rate": 1e-3, "verbose": False}

    if args.classifier_type == "Linear":
        classifier_class = LinearClassifier
    elif args.classifier_type == "MLP":
        classifier_class = MLPClassifier
        classifier_args.update(
            {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers}
        )
    else:
        classifier_class = RNNClassifier
        classifier_args.update(
            {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers}
        )

    results = {
        "layer": layer,
        "model": args.model,
        "attacks": args.attacks.split("+"),
        "source": args.source,
        "classifier_type": args.classifier_type,
        "classifier_args": classifier_args,
        "train_label": args.train,
    }

    print(f"Training layer classifier for layer {layer}...")
    benign_data = data_cache["benign"][layer]["embedding"]

    classifier_args["input_dim"] = benign_data["x"].shape[-1]
    layer_results, layer_classifiers, split_bases = train_and_evaluate(
        benign_data["x"],
        benign_data["y"],
        benign_data["y_pred"],
        benign_data["bases"],
        classifier_class,
        classifier_args,
        n_splits=args.n_splits,
        early_stopping=args.early_stopping,
        patience=args.patience,
        num_epochs=args.num_epochs,
        train_on=args.train,
    )

    results["layer_classifier"] = layer_results

    # Evaluate attacks
    results["layer_classifier"]["attacks"] = {}
    for attack in args.attacks.split("+"):
        if attack == "benign":
            continue

        attack_data = data_cache[attack][layer]["embedding"]
        attack_results = {"attack": attack, "fold_results": []}

        for fold_idx, clf in enumerate(layer_classifiers):
            # Get the base indices for this fold
            fold_bases = split_bases[fold_idx]

            # Create masks for matching bases in attack data
            attack_bases = attack_data["bases"]
            train_mask = np.isin(attack_bases, fold_bases["train"])
            test_mask = np.isin(attack_bases, fold_bases["test"])

            # Split attack data according to the original base splits
            x_train_attack = attack_data["x"][train_mask]
            y_train_attack = attack_data["y"][train_mask]
            y_pred_train_attack = attack_data["y_pred"][train_mask]

            x_test_attack = attack_data["x"][test_mask]
            y_test_attack = attack_data["y"][test_mask]
            y_pred_test_attack = attack_data["y_pred"][test_mask]

            # Evaluate on both train and test sets
            best_threshold = layer_results["metadatas"][fold_idx]["best_threshold"]
            train_eval_true = clf.evaluate(
                x_train_attack, y_train_attack, threshold=best_threshold
            )
            train_eval_pred = clf.evaluate(
                x_train_attack, y_pred_train_attack, threshold=best_threshold
            )

            test_eval_true = clf.evaluate(
                x_test_attack, y_test_attack, threshold=best_threshold
            )
            test_eval_pred = clf.evaluate(
                x_test_attack, y_pred_test_attack, threshold=best_threshold
            )

            # Transfer evaluation for y_pred != y_true
            x_train_transfer = x_train_attack[y_train_attack != y_pred_train_attack]
            y_train_transfer = y_train_attack[y_train_attack != y_pred_train_attack]
            y_pred_train_transfer = y_pred_train_attack[
                y_train_attack != y_pred_train_attack
            ]
            x_test_transfer = x_test_attack[y_test_attack != y_pred_test_attack]
            y_test_transfer = y_test_attack[y_test_attack != y_pred_test_attack]
            y_pred_test_transfer = y_pred_test_attack[
                y_test_attack != y_pred_test_attack
            ]

            # Split transfer into harmful and harmless
            x_test_transfer_harmful = x_test_transfer[y_test_transfer == 1]
            y_test_transfer_harmful = y_test_transfer[y_test_transfer == 1]
            y_pred_test_transfer_harmful = y_pred_test_transfer[y_test_transfer == 1]
            x_test_transfer_harmless = x_test_transfer[y_test_transfer == 0]
            y_test_transfer_harmless = y_test_transfer[y_test_transfer == 0]
            y_pred_test_transfer_harmless = y_pred_test_transfer[y_test_transfer == 0]

            transfer_train_eval_true = clf.evaluate(
                x_train_transfer, y_train_transfer, threshold=best_threshold
            )
            transfer_train_eval_pred = clf.evaluate(
                x_train_transfer, y_pred_train_transfer, threshold=best_threshold
            )

            transfer_test_eval_true = clf.evaluate(
                x_test_transfer, y_test_transfer, threshold=best_threshold
            )
            transfer_test_eval_pred = clf.evaluate(
                x_test_transfer, y_pred_test_transfer, threshold=best_threshold
            )

            transfer_test_eval_true_harmful = clf.evaluate(
                x_test_transfer_harmful,
                y_test_transfer_harmful,
                threshold=best_threshold,
            )
            transfer_test_eval_pred_harmful = clf.evaluate(
                x_test_transfer_harmful,
                y_pred_test_transfer_harmful,
                threshold=best_threshold,
            )

            transfer_test_eval_true_harmless = clf.evaluate(
                x_test_transfer_harmless,
                y_test_transfer_harmless,
                threshold=best_threshold,
            )
            transfer_test_eval_pred_harmless = clf.evaluate(
                x_test_transfer_harmless,
                y_pred_test_transfer_harmless,
                threshold=best_threshold,
            )

            attack_results["fold_results"].append(
                {
                    "train": {
                        "true_acc": train_eval_true[0],
                        "true_f1": train_eval_true[1],
                        "agreement_llm": train_eval_pred[0],
                        "agreement_llm_f1": train_eval_pred[1],
                        "num_samples": len(x_train_attack),
                    },
                    "test": {
                        "true_acc": test_eval_true[0],
                        "true_f1": test_eval_true[1],
                        "agreement_llm": test_eval_pred[0],
                        "agreement_llm_f1": test_eval_pred[1],
                        "num_samples": len(x_test_attack),
                    },
                    "transfer": {
                        "train": {
                            "true_acc": transfer_train_eval_true[0],
                            "true_f1": transfer_train_eval_true[1],
                            "agreement_llm": transfer_train_eval_pred[0],
                            "agreement_llm_f1": transfer_train_eval_pred[1],
                            "num_samples": len(x_train_transfer),
                        },
                        "test": {
                            "true_acc": transfer_test_eval_true[0],
                            "true_f1": transfer_test_eval_true[1],
                            "agreement_llm": transfer_test_eval_pred[0],
                            "agreement_llm_f1": transfer_test_eval_pred[1],
                            "num_samples": len(x_test_transfer),
                        },
                        "harmful": {
                            "true_acc": transfer_test_eval_true_harmful[0],
                            "true_f1": transfer_test_eval_true_harmful[1],
                            "agreement_llm": transfer_test_eval_pred_harmful[0],
                            "agreement_llm_f1": transfer_test_eval_pred_harmful[1],
                            "num_samples": len(x_test_transfer_harmful),
                        },
                        "harmless": {
                            "true_acc": transfer_test_eval_true_harmless[0],
                            "true_f1": transfer_test_eval_true_harmless[1],
                            "agreement_llm": transfer_test_eval_pred_harmless[0],
                            "agreement_llm_f1": transfer_test_eval_pred_harmless[1],
                            "num_samples": len(x_test_transfer_harmless),
                        },
                    },
                }
            )

        results["layer_classifier"]["attacks"][attack] = attack_results

    if args.cross_source != "":
        cross_source_data = data_cache["cross_source"]["benign"][layer]["embedding"]
        cross_source_results = {"source": args.cross_source, "fold_results": []}
        for fold_idx, clf in enumerate(layer_classifiers):
            x_cross_source = cross_source_data["x"]
            y_cross_source = cross_source_data["y"]
            y_pred_cross_source = cross_source_data["y_pred"]

            # Evaluate on the combined set
            best_threshold = layer_results["metadatas"][fold_idx]["best_threshold"]
            eval_true = clf.evaluate(
                x_cross_source, y_cross_source, threshold=best_threshold
            )
            eval_pred = clf.evaluate(
                x_cross_source, y_pred_cross_source, threshold=best_threshold
            )

            cross_source_results["fold_results"].append(
                {
                    "true_acc": eval_true[0],
                    "true_f1": eval_true[1],
                    "agreement_llm": eval_pred[0],
                    "agreement_llm_f1": eval_pred[1],
                    "num_samples": len(x_cross_source),
                }
            )

        results["layer_classifier"]["cross_source"] = cross_source_results

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = get_args()

    print("Loading data...")
    # Load all data once
    mn = ParquetManager()
    data_cache = load_all_data(
        mn,
        args.model,
        args.start_layer,
        args.end_layer,
        attacks=args.attacks.split("+"),
        source=args.source,
        cross_source=args.cross_source,
    )

    print("Data loaded.")

    if args.sequential:
        layer_results = [
            process_layer(args, layer, data_cache)
            for layer in range(args.start_layer, args.end_layer + 1)
        ]
    else:
        # Process layers in parallel
        with mp.Pool(processes=args.end_layer - args.start_layer + 1) as pool:
            layer_results = pool.starmap(
                process_layer,
                [
                    (args, layer, data_cache)
                    for layer in range(args.start_layer, args.end_layer + 1)
                ],
            )

    combined_results = layer_results
    output_filename = (os.path.join(args.output_dir, f"clf_analysis_{args.model}_{args.source}_{args.classifier_type}_layer{args.start_layer:02d}-{args.end_layer:02d}_{args.train}_{args.attacks}.json"))

    with open(output_filename, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results saved to {output_filename}")
