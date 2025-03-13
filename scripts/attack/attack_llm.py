import json
import os

import torch
from clfextract.configs import set_config
from clfextract.evaluators import PipelineEvaluator
from gcg_llm import GCGConfig, run_gcg
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

if __name__ == "__main__":
    config = set_config()

    print(config)

    tokenizer, model = config.exp.tokenizer, config.exp.model
    start, end = config.exp.start, config.exp.end

    dataset = config.exp.dataset
    records = []
    gcg_config = GCGConfig(
        num_steps=int(config.misc.num_steps),
        search_width=int(config.misc.search_width),
        topk=int(config.misc.topk),
    )
    metadata = {
        "attack": "gcg",
        "dataset": os.path.basename(config.exp.dataset_path),
        "model_path": config.exp.model_path,
        "attack_config": {
            "num_steps": gcg_config.num_steps,
            "search_width": gcg_config.search_width,
            "topk": gcg_config.topk,
        },
        "clf": None,
    }

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

    evaluator = PipelineEvaluator(
        config.exp.model,
        config.exp.tokenizer,
        config.prompt_manager,
        clf_pipeline,
    )

    for i in tqdm(range(config.exp.start, config.exp.end)):
        base = dataset["base"][i]
        target = dataset["target"][i]
        messages = [{"role": "user", "content": base}]
        result = run_gcg(model, tokenizer, messages, target, gcg_config)

        best_string, best_loss = result.best_string, result.best_loss
        strings, losses = result.strings, result.losses

        y_llm_base = evaluator([{"base": base, "attack": ""}])

        # Every 100 steps until the num_steps
        eval_steps = list(range(100, gcg_config.num_steps + 1, 100))

        step_results = []
        for step in eval_steps:
            # argmin of losses[:step]
            best_string_step = strings[losses.index(min(losses[:step]))]
            best_loss_step = min(losses[:step])

            y_llm = evaluator([{"base": base, "attack": best_string_step}])
            success = abs(y_llm - y_llm_base)
            output = evaluator.logger[-1].get("output", None)
            step_result = {
                "step": step,
                "attack": best_string_step,
                "loss": best_loss_step,
                "y_llm_base": y_llm_base,
                "y_llm": y_llm,
                "success": success,
                "output": output,
            }
            step_results.append(step_result)

        # Final evaluation
        y_llm = evaluator([{"base": base, "attack": best_string}])
        success = abs(y_llm - y_llm_base)
        output = evaluator.logger[-1].get("output", None)
        records.append(
            {
                "base": base,
                "target": target,
                "attack": best_string,
                "loss": best_loss,
                "y_llm_base": y_llm_base,
                "y_llm": y_llm,
                "step_results": step_results,
                "success": success,
                "output": output,
                "append_strat": "suffix",
                "attacks": strings,
                "losses": losses,
            }
        )

    results = {"metadata": metadata, "records": records}

    with open(
        f'llama2_gcg_{gcg_config.num_steps}steps_{gcg_config.search_width}search_width_{gcg_config.topk}topk_{config.exp.dataset_path.split("/")[-1].replace(".json","")}_{start:03d}_{end:03d}.json',
        "w",
    ) as f:
        json.dump(results, f, indent=4)
