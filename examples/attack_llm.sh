dataset_dir="/workspace/data/bases/"
output_path="/workspace/results/"

# Variables
model_path="meta-llama/Llama-2-7B-chat-hf"
dataset_name="advbench_harmful.json"
start=0
end=1

python3 /workspace/scripts/attack/attack_llm.py --model_path $model \
                    --dataset_path $dataset_dir$dataset_name \
                    --output_path $output_path \
                    --start $start \
                    --end $end \
                    --half \
                    --append_strat suffix \
                    --num_steps 250 \
                    --search_width 512 \
                    --topk 512
