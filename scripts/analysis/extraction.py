import gc
import os

import polars as pl
import torch
import numpy as np
from clfextract.configs import set_config
from clfextract.evaluators import (
    EnsembleEvaluator,
    PipelineEvaluator,
    StringMatchEvaluator,
)
from clfextract.lenses import EmbeddingLens
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def map_y_pred(y_pred):
    if not isinstance(y_pred, str):
        return None
    if y_pred.lower() == "normal":
        return False
    elif y_pred.lower() == "rejection":
        return True
    else:
        return None


def rows_from_item(item: dict, lens_key: str, y_true: int, attack: str, source: str):
    assert lens_key in item
    data = item[lens_key]  # numpy array
    if len(data.shape) == 2:
        # Add a dimension at 1st shape to make it 3D
        data = data[:, np.newaxis, :]
    rows = []
    if "key" in lens_key or "value" in lens_key:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                num_positions = data.shape[2]
                row = {
                    "base": item["base"],
                    "attack": attack,
                    "output": item["output"],
                    "source": source,
                    "layer": i,
                    "head": j,
                    "x": data[i, j, :, :].tobytes(),
                    "num_positions": num_positions,
                    "y_true": bool(y_true),
                    "y_pred_advbench": bool(item["labels"][0]),
                    "y_pred_protectai": map_y_pred(item["labels"][1]),
                }
                rows.append(row)
        return rows

    elif "embedding" == lens_key:
        for i in range(data.shape[0]):
            num_positions = data.shape[1]
            row = {
                "base": item["base"],
                "attack": attack,
                "output": item["output"],
                "source": source,
                "layer": i,
                "x": data[i, :, :].tobytes(),
                "num_positions": num_positions,
                "y_true": bool(y_true),
                "y_pred_advbench": bool(item["labels"][0]),
                "y_pred_protectai": map_y_pred(item["labels"][1]),
            }
            rows.append(row)
        return rows


def create_dataset_from_lens(evaluator, lens_name, y_true, attack, source):
    rows = []
    for i, item in enumerate(evaluator.logger):
        data = item[lens_name].squeeze()  # numpy array
        rows.extend(rows_from_item(item, lens_name, y_true[i], attack, source))

    return pl.DataFrame(rows)


def main():
    config = set_config()
    if "label" in config.exp.dataset.columns:
        y_true = config.exp.dataset["label"]
    else:
        y_true = (
            1
            if "harmful" in config.exp.dataset_path
            or "toxic" in config.exp.dataset_path
            else 0
        )
        y_true = [y_true] * len(config.exp.dataset)
    print(config)

    # TODO put exception to config
    if "llama-2" in config.exp.model_path.lower():
        model_tag = "llama2"
    elif "llama-3" in config.exp.model_path.lower():
        model_tag = "llama3"
    elif "gemma-2" in config.exp.model_path.lower():
        model_tag = "gemma2"
    elif "gemma-7b" in config.exp.model_path.lower():
        model_tag = "gemma1"
    elif "qwen2" in config.exp.model_path.lower():
        model_tag = "qwen2"
    elif "mistral" in config.exp.model_path.lower():
        model_tag = "mistral"
    elif "zephyr_rmu" in config.exp.model_path.lower():
        model_tag = "zephyrrmu"
    else:
        model_tag = config.exp.model_path.lower().split("/")[-1]
    config.exp.tokenizer.padding_side = "left"

    lenses = {}

    lens_name = "embedding"
    lens = EmbeddingLens(
        config.exp.model,
        config.exp.tokenizer,
        positions=[-1],
    )

    lenses = {"embedding": lens}

    heuristic_evaluator = StringMatchEvaluator(
        config.exp.model, config.exp.tokenizer, config.prompt_manager
    )

    clf_tokenizer = AutoTokenizer.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )

    clf_pipeline = pipeline(
        "text-classification",
        model=clf_model,
        tokenizer=clf_tokenizer,
        truncation=True,
        max_length=512,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    clf_evaluator = PipelineEvaluator(
        config.exp.model,
        config.exp.tokenizer,
        config.prompt_manager,
        clf_pipeline,
    )

    evaluator = EnsembleEvaluator([heuristic_evaluator, clf_evaluator], lenses=lenses)

    df = config.exp.dataset

    if "attack" in df.columns:
        df = df.dropna(subset=["base", "attack"])
    else:
        print("Warning: 'attack' column not found. Using empty string as default.")
        df = df.dropna(subset=["base"])
        df["attack"] = ""

    start = max(config.exp.start, 0)
    end = min(config.exp.end, len(df))

    evaluator(df[start:end].to_dict(orient="records"))

    data_name = os.path.splitext(os.path.basename(config.exp.dataset_path))[0]
    # Remove model name from data name
    data_name = data_name.replace(model_tag, "")
    attack_tag = config.misc.attack
    source = config.misc.source

    # Remove the model and tokenizer from memory
    del config.exp.model
    del config.exp.tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    data_dir = os.getenv("DATA_DIR", "./")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for lens_name in ["embedding"]:
        output_file = data_dir + f"{data_name}_{model_tag}_{lens_name}_last.parquet"
        output_file = output_file.replace("__", "_")
        df = create_dataset_from_lens(evaluator, lens_name, y_true, attack_tag, source)
        df.write_parquet(
            output_file,
            compression="zstd",
            compression_level=10,
        )
    return


if __name__ == "__main__":
    main()
