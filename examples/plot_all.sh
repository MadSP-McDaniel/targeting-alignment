viz_dir=/workspace/figures
results_dir=/workspace/data/results
scripts_dir=/workspace/scripts/plot

# Create the directory if it does not exist
mkdir -p $viz_dir

# Run the visualization scripts
python3 $scripts_dir/plot_metadatas.py -i $results_dir/metadatas.json -o $viz_dir/metadatas

models_lists=("llama2 qwen2 gemma1 gemma2 granite")
# models_lists=("llama2 qwen2 gemma1 gemma2 granite" "llama3 mistral zephyrrmu") # Include other models

for list_models in "${models_lists[@]}"; do
    if [[ $list_models == "llama2 qwen2 gemma1 gemma2 granite" ]]; then
        viz_dir=/workspace/figures/main_models
    else
        viz_dir=/workspace/figures/other_models
    fi

    python3 $scripts_dir/plot_clf.py -i $results_dir/clf_analysis.json -o $viz_dir/clf_analysis --models $list_models

    python3 $scripts_dir/plot_subspace.py -i $results_dir/space_analysis.json -o $viz_dir/subspace --models $list_models

    # Transfers
    datasets=("advbench_harmful.json or-bench_harmful.json" "advbench_harmless.json or-bench_harmless.json")


    for dataset in "${datasets[@]}"; do

        if [[ $dataset == *"harmful"* ]]; then
            python3 $scripts_dir/plot_transfer.py -i $results_dir/transfers.json -o $viz_dir/transfer --datasets $dataset --models $list_models --dataset_name harmful
        else
            python3 $scripts_dir/plot_transfer.py -i $results_dir/transfers.json -o $viz_dir/transfer --datasets $dataset --models $list_models --dataset_name harmless

        fi
    done

done

