#!/bin/bash

wandb enabled

# torchrun --nproc_per_node 4 run_2_pretrain_projectors.py \
#     --task=fineweb \
#     --n_mixtures=12 \
#     --per_device_batch_size=1 \
#     --gradient_accumulation_steps=1


torchrun --nproc_per_node 4 run_2_pretrain_projectors.py \
    --task=fineweb \
    --n_mixtures=1 \
    --per_device_batch_size=1 \
    --gradient_accumulation_steps=1
