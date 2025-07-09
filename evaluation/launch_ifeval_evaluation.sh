#!/bin/bash
# cd to the checklist_finetuning directory root
checklist_finetuning_dir=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $checklist_finetuning_dir
model=$1
python run_req_eval.py --benchmark  IFEval --method direct --llm  $model
formatted_path=$(echo "$model" | tr '/.*' '_')

outdir=ifeval_${formatted_path}
conda activate vllm

cd ${checklist_finetuning_dir}/google-research/
mkdir -p ${checklist_finetuning_dir}/IFEval_results/${outdir}/
python3 -m instruction_following_eval.evaluation_main \
--input_data=${checklist_finetuning_dir}/google-research/instruction_following_eval/data/input_data.jsonl \
--input_response_data=${checklist_finetuning_dir}/${formatted_path}_response_data.jsonl \
--output_dir=${checklist_finetuning_dir}/IFEval_results/${outdir}/