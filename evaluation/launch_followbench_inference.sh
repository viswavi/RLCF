#!/bin/bash
export LITELLM_API_KEY=$2
model=$1

MODEL_NAME=$(echo "$model" | tr '/' '_')
echo "STARTING $model FOLLOWBENCH EVALUATION"
python run_req_eval.py --benchmark FollowBench --method direct --llm "$model" --num-batches 10
formatted_path=$(echo "$model" | tr '/.-' '_')

eval_dir="$(dirname "$(readlink -f "$0")")"
python ${eval_dir}/summarize_followbench_results.py $formatted_path

