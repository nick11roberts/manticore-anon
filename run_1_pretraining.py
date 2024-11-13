import sys
import torch
import fire
import yaml
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image, ImageOps
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import (
    GPTNeoForCausalLM,
    GPTNeoForTokenClassification,
    GPTNeoForSequenceClassification,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoConfig
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from models.manticore import (
    ManticoreConfig,
    ManticoreForTokenClassification,
    ManticoreForLEGO,
    ManticoreForICLRegression,
    ManticoreForCausalLM,
    ManticoreForSequenceClassification,
)
from models.mamba_custom import (
    MambaConfigHF,
    MambaForCausalLM,
    MambaForTokenClassification,
    MambaForSequenceClassification,
)
from models.manticore_mad import (
    ManticoreConfig as ManticoreMadConfig,
    ManticoreMadForTokenClassification,
    ManticoreMadForCausalLM,
)
from callbacks.wandb import WandbAlphasCallback
from data.lego.lego_data import lego_dataset
from data.icl_regression.data import get_regression_dataset, RegressionIterDataset
from data.natural_language.eli5 import get_eli5_dataset
from data.natural_language.ptb import get_ptb_dataset
from data.natural_language.alpaca import get_alpaca_dataset
from data.prepare_lra import get_lra_dataset
from data.mad.get_mad import get_mad
from nas import (
    MixedOptimizer, MixedDataset, 
    create_manticore_mixed_optimizer, MixedOptTrainer
)
from data.automata.automata import AutomatonDataset

from run_configs import get_experiment_config

import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# device = "cuda"


def main(
    task: str = "mad:memorization",
    model_type: str = "manticore",
    d_model=128,
    n_mixtures=4,
    use_projectors=True,
    pretrain=True,
    load_projectors=False,
    epochs=10,
    device="cuda",
    disable_tqdm=False,
    alpha_mamba=None,
    projector_type="default",
    projector_gpt=False,
    hp_search=False,
    use_mixed_optimizer=True,
    ## TODO: more args for controlling mixed optimizer?
):

    task_split = task.split(":")
    if len(task_split) == 1:
        task_split.append(None)
    task_type, task_name = task_split
    print(task_type, task_name)

    (
        model_clses,
        model_kwargs,
        hps,
        trainer_kwargs,
        train_dataset,
        eval_dataset,
    ) = get_experiment_config(
        task_type,
        task_name,
        ntrain=50_000,
        d_model=d_model,
        n_mixtures=n_mixtures,
        epochs=epochs,
        model_cls=model_type, # EDIT
    )
    print("###### IN MAIN DATASET LENS: ", len(train_dataset), len(eval_dataset))
    def model_init(trial): # same model for every trial; EDIT
        if model_type == "manticore":
            config = ManticoreConfig(
                d_model=d_model,
                n_mixtures=n_mixtures,
                use_projectors=use_projectors,
                alpha_mamba=alpha_mamba,
                projector_type=projector_type,
                projector_gpt=projector_gpt,
            )
    
            print(config)
    
            model = model_clses["manticore"](
                config=config,
                pretrain=pretrain,
                **model_kwargs,
            ).to(device=device)
        elif model_type == "manticore_mad": # EDIT
            config = ManticoreMadConfig(
                d_model=d_model,
                n_mixtures=n_mixtures,
                use_projectors=use_projectors,
                projector_type=projector_type,
            )
    
            model = model_clses["manticore_mad"](
                config=config,
                pretrain=pretrain,
                **model_kwargs,
            ).to(device)#.to(torch.float16) # is there a better way to handle this? attn modules from flash-attn require float16
        elif model_type in ["mad_mambaformer", "mad_striped_mh_hyena"]: # EDIT
            model = model_clses[model_type](
                **model_kwargs
            )
        elif model_type == "mamba":
            config = MambaConfigHF(model_kwargs["mamba"].config)
            if "num_classes" in model_kwargs.keys():
                config.num_labels = model_kwargs.pop("num_classes", None)
            model = model_clses["mamba"](config=config).to(device=device)
        elif model_type == "gptneo":
            config = model_kwargs["gptneo"].config
            if "num_classes" in model_kwargs.keys():
                config.num_labels = model_kwargs.pop("num_classes", None)
            model = model_clses["gptneo"](config=config).to(device=device)
        else:
            raise NotImplementedError
            
        if task_type == "lra":
            if trainer_kwargs.get("tokenizer") is not None:
                model.config.pad_token_id = 0  # 50256
            else:
                model.config.pad_token_id = 50256
        print(model)
        # print the parameter count
        num_params = sum(p.numel() for p in model.parameters())
        print("num_params", num_params)
        if model_type == "manticore":
            # model.disable_alphas()  # ❌ disable alphas
            model.enable_alphas()  # ✅ enable alphas
            model.enable_pretraining()  # ✅ enable projector and model learning
        return model # EDIT

    
    print(task)
    
    os.environ["WANDB_PROJECT"] =  "-".join(["pretrain", task_type, task_name, model_type])
    training_args = TrainingArguments(
        # output_dir="trash",
        output_dir = os.path.join("ckpts", "-".join(["pretrain", task_type, task_name, model_type])),
        # evaluation_strategy="steps",
        # eval_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=200_000,
        save_safetensors=False,
        report_to="wandb", 
        disable_tqdm=disable_tqdm,
        skip_memory_metrics=True,
        **hps,
    )
    if use_mixed_optimizer:
        ## TODO?: move the MixedOptTrainer args somewhere else instead of hardcoding
        trainer = MixedOptTrainer(
            arch_lr = 1e-1,
            alternating=False, ## alternating doesnt work with dataset 
            steps_per_opt=None,
            ##
            model=None,
            model_init=model_init, #EDIT
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[WandbAlphasCallback(freq=1)],
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(
            # model=model,
            model=None,
            model_init=model_init, #EDIT
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[WandbAlphasCallback(freq=1)],
            **trainer_kwargs,
        )
    # EDIT
    if hp_search: # SO FAR THE HP SEARCH CONFIGURATION IS ONLY SET SPECIFICALLY FOR ICL REGRESSION TASKS
        print("\n\n##### STARTING ASHA HPO ######")
        mode = "min"
        # max_t = 500_000 # mins or eval iterations; for ICLRegr
        # grace_period = 100_000
        max_t = 200 # for MAD
        grace_period = 40
        time_attr = "training_iteration"
        scheduler = ASHAScheduler(
            time_attr=time_attr,
            max_t=max_t,
            metric = "eval_loss",
            mode = "min",
            grace_period=grace_period,
        )
        def get_hpo_metric(target_metric: str, metrics: dict):
            return metrics[target_metric]
        reporter = CLIReporter(
            # parameter_columns=["learning_rate", "max_grad_norm", ], # FOR ICLRegr
            parameter_columns=["learning_rate", "per_device_train_batch_size", "weight_decay"],
            metric_columns=["train_loss", "eval_loss", "training_iteration"],
            max_progress_rows=9,
            max_report_frequency=9,
        )   
        param_space = {
            # "learning_rate": tune.qloguniform(5e-5, 4e-4, 1e-5), # FOR ICLRegr
            # "max_grad_norm": tune.choice([5., 10., 50.]),
            # "lr_scheduler_type": "cosine", # mostly linear underperforms
            "learning_rate": tune.choice([1e-4, 5e-4, 1e-3]),
            "per_device_train_batch_size": tune.choice([128, 64, 16, 4]), 
            "weight_decay": tune.choice([0.0, 0.1]),
        }
        
        best_run = trainer.hyperparameter_search(
                hp_space=lambda _: param_space,
                backend="ray",
                # n_trials=12, # FOR ICLRegr
                n_trials= 16,
                scheduler=scheduler,
                keep_checkpoints_num=None,
                checkpoint_score_attr="max-" + "eval_loss", # rank in decreasing order
                progress_reporter=reporter,
                resources_per_trial={"cpu": 1, "gpu": 1},
                name=task + "-"+model_type, # os.environ["WANDB_RUN_GROUP"],
                max_failures=999, # tolerate OOM
                direction="minimize",
                compute_objective=partial(get_hpo_metric, "eval_loss" ),
                # resume=args.resume
        )
        best_hp = best_run.hyperparameters
        print(best_hp)
    else:
        print("\n\n##### STARTING NORMAL TRAINING ######")
        trainer.train()

    # if model_type == "manticore":
    #     model.enable_alphas()  # ✅ enable alphas
    #     model.enable_pretraining()  # ✅ enable projector and model learning
    # training_args = TrainingArguments(
    #     output_dir="trash",
    #     # evaluation_strategy="steps",
    #     # eval_steps=1000,
    #     evaluation_strategy="epoch",
    #     save_strategy="steps",
    #     save_steps=5000,
    #     save_safetensors=False,
    #     report_to="wandb",
    #     disable_tqdm=disable_tqdm,
    #     **hps,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     callbacks=[WandbAlphasCallback(freq=1)],
    #     **trainer_kwargs,
    # )
    # trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
