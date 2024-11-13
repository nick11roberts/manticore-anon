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

# task=ptb
# basedir=manticore_logs/natural_language/${task}/alpha_sweep
# mkdir -p ${basedir}

# for alpha_mamba in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do 
#     python -u run_2_finetuning.py \
#         --task=${task} \
#         --model_type=manticore \
#         --n_mixtures=1 \
#         --alpha_mamba=${alpha_mamba} |& tee -a ${basedir}/manticore_${alpha_mamba}.log 
# done 


d_model=128
epochs=200

mkdir -p manticore_logs/mad/

for n_mixtures in 1 2 4 8 16 
do

    for task in fuzzy-in-context-recall in-context-recall memorization noisy-in-context-recall selective-copying
    do 
        basedir=manticore_logs/mad/${task}/epochs_${epochs}/n_mixtures_${n_mixtures}/${d_model}/
        mkdir -p ${basedir}

        # Run all models in parallel 
        #pids=
        for model_type in manticore mamba gptneo
        do 

            python -u run_1_pretraining.py \
                --n_mixtures=${n_mixtures} \
                --d_model=${d_model} \
                --epochs=${epochs} \
                --model_type=${model_type} \
                --task=mad:${task} |& tee -a ${basedir}/${model_type}.log # &

            #pids+=" $!"
        done
        #wait $pids || { echo "there were errors" >&2; exit 1; }

    done
done


