#!/bin/bash

epochs=200
n_mixtures=2

for task in fuzzy-in-context-recall in-context-recall memorization noisy-in-context-recall selective-copying
do 
    basedir=manticore_logs/mad_programming/${task}/epochs_${epochs}/n_mixtures_${n_mixtures}/
    mkdir -p ${basedir}

    python -u run_3_programming.py \
        --task=mad:selective-copying \
        --n_mixtures=${n_mixtures} \
        --use_projectors=False \
        --load_projectors=False \
        --arch_lr=0.01 \
        --epochs=${epochs} |& tee -a ${basedir}/programming.log 

done