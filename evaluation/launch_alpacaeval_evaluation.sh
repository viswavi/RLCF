#!/bin/bash
model_outputs=/home/vijayv/compose-instruct/$1
alpaca_eval --model_outputs ${model_outputs} --annotators_config 'alpaca_eval_gpt4_turbo_fn'