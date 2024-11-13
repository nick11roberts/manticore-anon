#!/bin/bash

wandb enabled

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_2_pretrain_projectors.py \
    --task=fineweb \
    --n_mixtures=12 \
    --per_device_batch_size=1 \
    --gradient_accumulation_steps=1

# python run_2_pretrain_projectors.py \
#     --task=fineweb \
#     --n_mixtures=24 \
#     --per_device_batch_size=4 \
#     --gradient_accumulation_steps=1
