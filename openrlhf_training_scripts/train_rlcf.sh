#!/bin/bash

set -x

savepath=$1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${savepath} \
   --ckpt_path ${savepath} \
   --save_steps 32 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 1024 \
   --micro_train_batch_size 4 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 6 \
   --max_len 2048 \
   --zero_stage 3 \
   --learning_rate 3e-6 \
   --min_lr_ratio 0.75 \
   --beta 0.1 \
   --dataset viswavi/rlcf \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --save_hf_ckpt \
   --max_ckpt_num 1 \
   --gradient_checkpointing
EOF


deepspeed --module $training_commands