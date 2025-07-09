#!/bin/bash
export LITELLM_API_KEY=$2
export OPENAI_API_KEY=$LITELLM_API_KEY

model=$1

# cd to the checklist_finetuning directory root
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
python run_req_eval.py --benchmark AlpacaEval --method direct --llm $model