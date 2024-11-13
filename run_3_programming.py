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
from nas import MixedOptimizer, MixedDataset
from data.automata.automata import AutomatonDataset

from run_configs import get_experiment_config

import os
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
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
    arch_lr=0.01,
    alternating=False,
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

    model = model_init(None)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbAlphasCallback(freq=1)],
        **trainer_kwargs,
    )



    from transformers import get_linear_schedule_with_warmup
    from run_2_optimizer import MixedOptimizer, MixedScheduler

    batch_size = 8
    epochs = 1
    num_steps = ((len(train_dataset)) // batch_size) * epochs

    optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    p_arch = [p for name, p in model.named_parameters() if "alphas" in name]
    p_model = [p for name, p in model.named_parameters() if "alphas" not in name]

    lr = optim_kwargs.pop("lr", None)
    optim_model = optim_cls(p_model, lr=lr, **optim_kwargs)
    
    # print(lr)
    # quit()
    optim_arch = optim_cls(p_arch, lr=arch_lr, **optim_kwargs)

    optimizer = MixedOptimizer([optim_model, optim_arch], alternating=alternating)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )
    # scheduler = MixedScheduler(optimizer, [sched_model, sched_arch])
    optims = (optimizer, scheduler)



    
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbAlphasCallback(freq=1)],
        optimizers=optims,
        **trainer_kwargs,
    )
    # EDIT
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
