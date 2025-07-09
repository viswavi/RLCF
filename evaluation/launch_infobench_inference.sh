#!/bin/bash
export LITELLM_API_KEY=$2
export OPENAI_API_KEY=$LITELLM_API_KEY

model=$1
MODEL_NAME=$(echo "$model" | tr '/' '_')
echo "STARTING $model INFOBENCH EVALUATION"
python run_req_eval.py --benchmark InFoBench --method direct --llm "$model" --num-batches 10


