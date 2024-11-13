#!/bin/bash

wandb disabled

# d_model=32
# epochs=10

# mkdir -p manticore_logs/mad/

# for n_mixtures in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do

#     for task in fuzzy-in-context-recall in-context-recall memorization noisy-in-context-recall selective-copying
#     do 
#         basedir=manticore_logs/mad/${task}/epochs_${epochs}/n_mixtures_${n_mixtures}/${d_model}/
#         mkdir -p ${basedir}

#         # Run all models in parallel 
#         pids=
#         for model_type in manticore mamba gptneo
#         do 

#             python -u run_1_pretraining.py \
#                 --n_mixtures=${n_mixtures} \
#                 --d_model=${d_model} \
#                 --epochs=${epochs} \
#                 --model_type=${model_type} \
#                 --task=mad:${task} |& tee -a ${basedir}/${model_type}.log &

#             pids+=" $!"
#         done
#         wait $pids || { echo "there were errors" >&2; exit 1; }

#     done
# done


# d_model=128
# epochs=100

# mkdir -p manticore_logs/mad/

# for n_mixtures in 2 4 8 
# do

#     for task in fuzzy-in-context-recall in-context-recall memorization noisy-in-context-recall selective-copying
#     do 
#         basedir=manticore_logs/mad/${task}/epochs_${epochs}/n_mixtures_${n_mixtures}/${d_model}/
#         mkdir -p ${basedir}

#         # Run all models in parallel 
#         pids=
#         for model_type in manticore mamba gptneo
#         do 

#             python -u run_1_pretraining.py \
#                 --n_mixtures=${n_mixtures} \
#                 --d_model=${d_model} \
#                 --epochs=${epochs} \
#                 --model_type=${model_type} \
#                 --task=mad:${task} |& tee -a ${basedir}/${model_type}.log &

#             pids+=" $!"
#         done
#         wait $pids || { echo "there were errors" >&2; exit 1; }

#     done
# done

# task=memorization
# basedir=manticore_logs/mad/${task}/alpha_sweep
# mkdir -p ${basedir}

# for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     python run_1_pretraining.py --n_mixtures=1 --d_model=128 --epochs=200 --task=mad:${task} --model_type=manticore --alpha_mamba=${alpha_mamba} --use_projectors=True |& tee -a ${basedir}/manticore_projectors_${alpha_mamba}.log 
# done


# ############## Mixture sweep
# #task=memorization

# for task in memorization selective-copying
# do
#     basedir=manticore_logs/mad/${task}/alpha_sweep
#     mkdir -p ${basedir}

#     for projtype in LinearProjector LinearSkipProjector TwoLayerProjector TwoLayerSkipProjector
#     do
#         for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#         do
#             python run_1_pretraining.py --n_mixtures=1 --d_model=128 --epochs=200 --task=mad:${task} --model_type=manticore --alpha_mamba=${alpha_mamba} --use_projectors=True --projector_gpt=False --projector_type=${projtype} |& tee -a ${basedir}/manticore_projectors_gptFalse_${projtype}_${alpha_mamba}.log 

#             python run_1_pretraining.py --n_mixtures=1 --d_model=128 --epochs=200 --task=mad:${task} --model_type=manticore --alpha_mamba=${alpha_mamba} --use_projectors=True --projector_gpt=True --projector_type=${projtype} |& tee -a ${basedir}/manticore_projectors_gptTrue_${projtype}_${alpha_mamba}.log 
#         done
#     done 
# done
# ############## Mixture sweep




# wandb enabled

# for task in ptb alpaca
# do
#     basedir=manticore_logs/natural_language/${task}
#     mkdir -p ${basedir}
#     for model in manticore # mamba gptneo
#     do
#         python -u run_2_finetuning.py --task=${task} --model_type=${model} |& tee -a $ ${basedir}/${model}_finetune.log
#     done 
# done

# for task in ptb alpaca eli5
# do
#     basedir=manticore_logs/natural_language/${task}/alpha_sweep/stable
#     mkdir -p ${basedir}

#     for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do 
#         python -u run_2_finetuning.py \
#             --task=${task} \
#             --model_type=manticore \
#             --n_mixtures=1 \
#             --use_projectors=True \
#             --load_projectors=False \
#             --alpha_mamba=${alpha_mamba} |& tee -a ${basedir}/manticore_${alpha_mamba}.log 
#     done 
# done

# wandb enabled

# # THEN PRETRAIN ON 1B TOKENS OF FINE WEB
# python -u run_2_pretrain_projectors.py \
#     --task=fineweb \
#     --model_type=manticore \
#     --n_mixtures=12 \
#     --use_projectors=True 


# for task in alpaca eli5 ptb
# do
#     basedir=manticore_logs/natural_language/${task}/alpha_sweep/5_16
#     mkdir -p ${basedir}

#     for alpha_mamba in 0.0 0.5 1.0
#     do 
#         python -u run_2_finetuning_alphasearch.py \
#             --task=${task} \
#             --model_type=manticore \
#             --n_mixtures=1 \
#             --use_projectors=True \
#             --load_projectors=True \
#             --data_frac=1 \
#             --alpha_mamba=${alpha_mamba} |& tee -a ${basedir}/manticore_${alpha_mamba}.log 
#     done 
# done




# for task in mix
# do
#     basedir=manticore_logs/natural_language/${task}/alpha_sweep/5_17
#     mkdir -p ${basedir}

#     for alpha_mamba in 1.0 0.5 0.9 0.8 0.7 0.6 0.4 0.3 0.2 0.1 #0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do 
#         python -u run_2_finetuning_alphasearch.py \
#             --task=${task} \
#             --model_type=manticore \
#             --n_mixtures=1 \
#             --use_projectors=True \
#             --load_projectors=True \
#             --data_frac=1 \
#             --alpha_mamba=${alpha_mamba} |& tee -a ${basedir}/manticore_${alpha_mamba}.log 
#     done 
# done


# for task in mix:mamba mix:gptneo
# do
#     basedir=manticore_logs/natural_language/${task}/alpha_sweep/5_17
#     mkdir -p ${basedir}

#     for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do 
#         python -u run_2_finetuning_alphasearch.py \
#             --task=${task} \
#             --model_type=manticore \
#             --n_mixtures=1 \
#             --use_projectors=True \
#             --load_projectors=True \
#             --data_frac=1 \
#             --alpha_mamba=${alpha_mamba} |& tee -a ${basedir}/manticore_${alpha_mamba}.log 
#     done 
# done

# 2D sweep

# for task in mix mix:mamba mix:gptneo
# do
#     basedir=manticore_logs/natural_language/${task}/alpha_2D_sweep/5_17
#     mkdir -p ${basedir}

#     for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do 
#         for alpha_mamba2 in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#         do 
#             python -u run_2_finetuning_alphasearch.py \
#                 --task=${task} \
#                 --model_type=manticore \
#                 --n_mixtures=2 \
#                 --use_projectors=True \
#                 --load_projectors=True \
#                 --data_frac=1 \
#                 --alpha_mamba=${alpha_mamba} \
#                 --alpha_mamba2=${alpha_mamba2} \
#                 |& tee -a ${basedir}/manticore_${alpha_mamba}_${alpha_mamba2}.log 
#         done 
#     done 
# done


# for task in ptb alpaca eli5
# do
#     basedir=manticore_logs/natural_language/${task}/n_mixtures_sweep/5_16
#     mkdir -p ${basedir}

#     for arch_lr in 0.005
#     do

#         for n_mixtures in 1 2 4 12 
#         do 
#             python -u run_2_finetuning.py \
#                 --task=${task} \
#                 --model_type=manticore \
#                 --n_mixtures=${n_mixtures} \
#                 --use_projectors=True \
#                 --load_projectors=True \
#                 --data_frac=1 \
#                 --arch_lr=${arch_lr} \
#                 --alternating=False \
#                 |& tee -a ${basedir}/manticore_n_mixtures${n_mixtures}_archlr${arch_lr}.log 
#         done 
#     done
# done


###### FINAL SYNTHETICS

# One-block search
# python -u run_2_finetuning.py --task=mix --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=1 --arch_lr=0.05 
# {'eval_loss': 1.3635727167129517, 'eval_runtime': 17.9926, 'eval_samples_per_second': 34.959, 'eval_steps_per_second': 4.391, 'epoch': 1.0, 'mixtures.0.alphas[0]': 0.7289986, 'mixtures.0.alphas[1]': 0.2710014}

# Two-block search 
# 

# Without pretraining projectors 
python -u run_2_finetuning.py --task=ptb --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=False --data_frac=10 --arch_lr=0.05 

# Pretraining projectors 
python -u run_2_finetuning.py --task=ptb --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=10 --arch_lr=0.05 

##### NAS ablation
# DARTS (simultaneous)
# 1.3635727167129517
# python -u run_2_finetuning.py --task=mix --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=1 --arch_lr=0.05 

# *******
# DARTS (alternating) -- orignal "Single-level" DARTS
# 1.2854355573654175
# {'eval_loss': 1.2863030433654785, 'eval_runtime': 17.9721, 'eval_samples_per_second': 34.999, 'eval_steps_per_second': 4.396, 'epoch': 1.0, 'mixtures.0.alphas[0]': 0.8326327, 'mixtures.0.alphas[1]': 0.16736726}
# python -u run_2_finetuning.py --task=mix --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=1 --arch_lr=0.05 --alternating=True

# DASH (simultaneous)  -- original DASH
# 2.5968732833862305
# python -u run_2_finetuning.py --task=mix --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=1 --arch_lr=0.005 --dash=True --alternating=False

# DASH (alternating) 
# 2.589916944503784
# python -u run_2_finetuning.py --task=mix --model_type=manticore --n_mixtures=1 --use_projectors=True --load_projectors=True --data_frac=1 --arch_lr=0.005 --dash=True --alternating=True



