# RLCF

## Generating Checklists
In the "candidate-based checklist" method, first need to generate responses by smaller LMs. You will need access to a single GPU with at least 40GB of memory for this:

```
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-0.5B --top-p 0.9 --temperature 0.6 --num-batches 50
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-1.5B --top-p 0.9 --temperature 0.6 --num-batches 50
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-3B --top-p 0.9 --temperature 0.6 --num-batches 50
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-7B --top-p 0.9 --temperature 0.6 --num-batches 50
```

Then, combine these responses into a single file
```
python group_wildchat_llm_responses.py \
    --input-responses-dir wildchat_responses \
    --combined-responses-dir combined_wildchat_responses
```

You can instead skip the previous two steps and fetch a pre-generated `combined_wildchat_responses` directory from https://drive.google.com/file/d/1TK9Z2o5gpjJeXQkCjlT3KocBybV59uMt/view?usp=sharing.

Then, we can generate checklists using Qwen/Qwen2.5-72B-Instruct (this has been tested on nodes with 4 80GB H100s or 8 40GB A100s). This will take several hours of processing time - up to 2 days, depending on your hardware.
```
# help me write a bash script ranging from 1 to 8
for i in {0..7}
do
    python data/requirement_generation/write_requirements.py \
        --job-idx $1 \
        --num-jobs 8 \
        --batch-size=5000 \
        --combined-response-data combined_wildchat_responses \
        --out-dir combined_wildchat_requirements
done
```

Now, you will have a directory, in `combined_wildchat_requirements`, which contains requirements for all the promps in wildchat! You can instead skip this step and fetch pre-generated files from https://drive.google.com/file/d/1hkZUDEc_QiywfwH-tQSvrlDjV_zTOlfx/view?usp=sharing.

### Generate Response Pairs
```
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-7B-Instruct --top-p 0.95 -num-samples 2 --temperature 1.3 --num-batches 50
```
These will be written to a directory called `wildchat_responses`. You can instead skip this step and fetch a version of `wildchat_responses` with pre-generated files from https://drive.google.com/file/d/1TutNY6uBC-MByAcaLzwwX_ypXhXVsCud/view?usp=sharing.


## Generate Preference Dataset

(tested on nodes with 4 or 8 H100s, withe computation done in batches)
```
cd data
python construct_offline_preference_data.py \
    --requirements-dir combined_wildchat_requirements/ \
    --candidates-source wildchat \
    --inference-type vllm \
    --produce-numerical-answers \
    --add-universal-requirements \
    --wildchat-candidates-glob 'wildchat_responses/Qwen_Qwen2_5_7B_Instruct_[0,1].jsonl' \
    --out-dir combined_wildchat_requirements/preference_data \
    --batch-start-idx 0 \
    --batch-end-idx 56
```

And then combine the generated files:
python combine_jsons.py combined_wildchat_scores.json


## Generate verification code
```
cd data/requirement_generation
python write_code_batch.py \
    --requirements-dir combined_wildchat_requirements \
    --sglang-model-name Qwen/Qwen2.5-72B-Instruct \
    --out-file verifiers.jsonl \
    --batch-size 1000
```

This step has been tested on nodes with 4 80GB H100 GPUs or 8 40GB A100 GPUs, and takes several hours. You can instead skip this step and fetch pre-generated files from https://drive.google.com/file/d/1F_dc5pexfPbMuW4kuM2b119tA3Nt1XrO/view?usp=sharing.

## Update checklist scores with code verification
```
mkdir -p rlcf_data_openrlhf
python generate_wildchat_openrlhf_dataset.py \
    --wildchat-rewards combined_wildchat_scores.json \
    --code-requirement-path verifiers.jsonl \
    --dataset-type rl \
    --output-file rlcf_data_openrlhf/train.jsonl
```

This file (`train.jsonl`) can then be used for training or evaluation.

## Train Model
This script has been tested on nodes with 4 or 8 H100s:
cd openrlhf_training_scripts
./train_rlcf.sh

## Benchmark Evaluation

### InFoBench
Set an
```
./launch_infobench_inference.sh  <trained_model_name> <openai or litellm key>
```

### FollowBench
```
./launch_followbench_inference.sh <trained_model_name> <openai or litellm key>
```

### IFEval
If you want to run evaluation on IFEval, you need to clone the google-research repo into the root of checklist_finetuning:
```
git clone https://github.com/google-research/google-research.git
```
Then, run ./launch_ifeval_evaluation.sh <trained_model_name>

### ArenaHard
If you want to run evaluation on IFEval, you need to clone the google-research repo into the root of checklist_finetuning:
```
git clone https://github.com/google-research/google-research.git
```
Then, run ./launch_ifeval_evaluation.sh <trained_model_name> <openai or litellm key>

# Dependencies
Install openrlhf from source:
```
conda create -n openrlhf python=3.10
pip install -r requirements.txt
```

# Cite
```
@misc{RLCF,
      title={Checklists Are Better Than Reward Models For Aligning Language Models},
      author={Vijay Viswanathan and Yanchao Sun and Shuang Ma and Xiang Kong and Meng Cao and Graham Neubig and Tongshuang Wu},
      year={2025},
      eprint={2507.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
