#!/bin/bash

python construct_offline_preference_data.py \
    --requirements-dir combined_wildchat_requirements/ \
    --candidates-source wildchat \
    --inference-type vllm \
    --produce-numerical-answers \
    --add-universal-requirements \
    --wildchat-candidates-glob 'wildchat_responses/Qwen_Qwen2_5_7B_Instruct_multi_sample_[0,7].jsonl' \
    --out-dir combined_wildchat_requirements/preference_data \
    --batch-start-idx 0 \
    --batch-end-idx 56