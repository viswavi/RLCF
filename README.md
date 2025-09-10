# RLCF: RL from Checklist Feedback

This is a software package to support automatically generating checklists/rubrics for instructions and using these checklists for grading responses for use in DPO. This code can easily be repurposed for use in online RL pipelines (e.g. GRPO).

We provide code for 5 main steps:

1. [Generating Checklists](https://github.com/viswavi/RLCF/blob/main/README.md#generating-checklists)
2. [Scoring Responses via Rubric-Grounded LM Judges](https://github.com/viswavi/RLCF/blob/main/README.md#scoring-responses-via-rubric-grounded-LM-judges)
3. [Scoring Responses via Rubric-Grounded Code Verifiers](https://github.com/viswavi/RLCF/blob/main/README.md#scoring-responses-via-rubric-grounded-code-verifiers)
4. [Train Model](https://github.com/viswavi/RLCF/blob/main/README.md#train-model)
5. [Evaluate on Benchmarks](benchmark-evaluation)


For your convenience, we've stored pre-computed results from each of these steps for [the particular setup described in our paper](https://arxiv.org/abs/2507.18624) (where we generated [checklists for WildChat](https://huggingface.co/datasets/viswavi/wildchecklists), generated [response pairs for each WildChat instruction]((https://drive.google.com/file/d/1TutNY6uBC-MByAcaLzwwX_ypXhXVsCud/view?usp=sharing)) using Qwen2.5-7B-Instruct, produced [scores for them](https://drive.google.com/file/d/1ixgWc72FSYoUq1RQ3_2uTGzNO7tPNxw-/view?usp=drive_link) using Qwen2.5-70B-Instruct as a judge, and generated [verification code](https://drive.google.com/file/d/1F_dc5pexfPbMuW4kuM2b119tA3Nt1XrO/view?usp=sharing) when applicable using Qwen2.5-70B-Instruct). The final result of this is a dataset suitable for offline RL: [viswavi/wildchecklists](https://huggingface.co/datasets/viswavi/wildchecklists).

We encourage you to generate similar data on policy for models you wish to train, and we present the code below to help you do that.


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

Now, you will have a directory, in `combined_wildchat_requirements`, which contains requirements for all the promps in wildchat! You can instead skip this step and fetch pre-generated files from https://drive.google.com/file/d/1hkZUDEc_QiywfwH-tQSvrlDjV_zTOlfx/view?usp=sharing or https://huggingface.co/datasets/viswavi/wildchecklists.

#### Generate Response Pairs
```
python -u run_req_eval.py --benchmark WildChat --method direct --llm Qwen/Qwen2.5-7B-Instruct --top-p 0.95 -num-samples 2 --temperature 1.3 --num-batches 50
```
These will be written to a directory called `wildchat_responses`. You can instead skip this step and fetch a version of `wildchat_responses` with pre-generated files from https://drive.google.com/file/d/1TutNY6uBC-MByAcaLzwwX_ypXhXVsCud/view?usp=sharing.

## Scoring Responses via Rubric-Grounded LM Judges

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

You can instead skip this step and fetch pre-generated files from https://drive.google.com/file/d/1ixgWc72FSYoUq1RQ3_2uTGzNO7tPNxw-/view?usp=drive_link.

## Scoring Responses via Rubric-Grounded Code Verifiers
#### Generate verification code
```
cd data/requirement_generation
python write_code_batch.py \
    --requirements-dir combined_wildchat_requirements \
    --sglang-model-name Qwen/Qwen2.5-72B-Instruct \
    --out-file verifiers.jsonl \
    --batch-size 1000
```

This step has been tested on nodes with 4 80GB H100 GPUs or 8 40GB A100 GPUs, and takes several hours. You can instead skip this step and fetch pre-generated files from https://drive.google.com/file/d/1F_dc5pexfPbMuW4kuM2b119tA3Nt1XrO/view?usp=sharing.

#### Update checklist scores with code verification
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
This script has been tested on nodes with 4 or 8 H100s/A100-80GBs:
```
cd openrlhf_training_scripts
./train_rlcf.sh
```

## Benchmark Evaluation

#### InFoBench
Set an
```
./launch_infobench_inference.sh  <trained_model_name> <openai or litellm key>
```

#### FollowBench
```
./launch_followbench_inference.sh <trained_model_name> <openai or litellm key>
```

#### IFEval
If you want to run evaluation on IFEval, you need to clone the google-research repo into the root of checklist_finetuning:
```
git clone https://github.com/google-research/google-research.git
```
Then, run ./launch_ifeval_evaluation.sh <trained_model_name>

#### ArenaHard
If you want to run evaluation on IFEval, you need to clone the google-research repo into the root of checklist_finetuning:
```
git clone https://github.com/google-research/google-research.git
```
Then, run ./launch_ifeval_evaluation.sh <trained_model_name> <openai or litellm key>

## Dependencies
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
      eprint={2507.18624},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
