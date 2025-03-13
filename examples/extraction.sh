#!/bin/bash
dataset_dir="/workspace/data/bases/"

models=("google/gemma-7b-it" "google/gemma-2-9b-it" "ibm-granite/granite-3.1-8b-instruct" "meta-llama/Llama-2-7b-chat-hf" "Qwen/Qwen2.5-7B-Instruct")
# models=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "cais/Zephyr_RMU") # Other Models

datasets=("advbench_harmful.json" "advbench_harmless.json" "or-bench_harmful.json" "or-bench_harmless.json")

start=0
end=1

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Get the source of the dataset if "advbench" or "or" in the name of the dataset
        if [[ $dataset == *"advbench"* ]]; then
            source="advbench"
        elif [[ $dataset == *"or-bench"* ]]; then
            source="or-bench"
        fi

        if [[ $dataset == *"gcg"* ]]; then
            attack="gcg"
        else
            attack="benign"
        fi

        python3 extraction.py --model_path $model \
                              --dataset_path $dataset_dir$dataset \
                              --append_strat $append_strat \
                              --start $start \
                              --end $end \
                              --half \
                              --source $source \
                              --attack $attack
    done
done