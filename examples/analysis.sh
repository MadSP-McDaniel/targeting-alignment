results_dir=/workspace/data/results
scripts_dir=/workspace/scripts

models=("gemma1" "gemma2" "granite" "llama2" "qwen2")
# models=("llama3" "mistral" "zephyrrmu") # Other Models

datasets=("advbench" "or-bench")

# Model-agnostic scripts
python3 $scripts_dir/metadata.py -o $results_dir || echo "Error running metadata.py" >&2
python3 $scripts_dir/space_analysis.py -o $results_dir || echo "Error running space_analysis.py" >&2

# List of models
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
    python3 $scripts_dir/clf_analysis.py -o $results_dir --model model || echo "Error running clf_analysis.py for model $model on dataset $dataset" >&2
    done
done