dataset_dir="/workspace/data/bases/"
output_path="/workspace/results/"

# Variables
model_path="meta-llama/Llama-2-7B-chat-hf"
dataset_name="advbench_harmful.json"
start=0
end=1
layers=(5 10 15 20 25 30) # Modify based on model

for layer in "${layers[@]}"; do
    python3 /workspace/scripts/attack/attack_clf.py --model_path $model_path \
                        --dataset_path $dataset_dir$dataset_name \
                        --output_path $output_path \
                        --start $start \
                        --end $end \
                        --current_layer $layer \
                        --half \
                        --num_steps 250 \
                        --search_width 512 \
                        --topk 512
done