import sys
import torch
import fire
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

from datasets import load_dataset

from mamba_ssm import MambaLMHeadModel

from models.partial_mamba import PartialRunnerForMambaLMHeadModel
from models.partial_gpt_neo import PartialRunnerForGPTNeoForCausalLM

# from based.models.gpt import GPTLMHeadModel
# from models.manticore import (
#     ManticoreConfig,
#     ManticoreForTokenClassification,
#     ManticoreForLEGO,
#     ManticoreForICLRegression,
#     ManticoreForCausalLM,
# )

from callbacks.wandb import WandbAlphasCallback
import wandb

from data.lego.lego_data import lego_dataset
from data.icl_regression.data import get_regression_dataset, RegressionIterDataset

from main import get_experiment_config
from models.partial_model_wrappers import PartialWrapperForICLRegression

def main(
    task: str = "regression:linear_regression",
    model_type:str = "mamba",
    # d_model=768,
    # n_mixtures=4,
    # use_projectors=True,
    # fixed_alphas=False,
    pretrain=False, # TODO
    device="cuda",
    log_wandb=True,
    disable_tqdm=True,
):

    start= time.time()
    
    if log_wandb:  # TODO: make wandb init not hardcoded
        wandb.init(project="manticore", entity="srguo", name=task + "-"+model_type)

    task_split = task.split(":")
    if len(task_split) == 1:
        task_split.append(None)
    task_type, task_name = task_split
    print(task_type, task_name)

    # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    # based = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m")

    _, model_kwargs, hps, trainer_kwargs, train_dataset, eval_dataset = (
        get_experiment_config(task_type, task_name, ntrain=50_000)
    )  # Hacky cannabilization of function from main.py

    if task_type=="regression":
        model_cls = PartialWrapperForICLRegression
    # TODO all other tasks
    else:
        raise NotImplementedError 
    
    if model_type=="mamba":
        model = PartialRunnerForMambaLMHeadModel(
            MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        )
    elif model_type=="gptneo":
        model = PartialRunnerForGPTNeoForCausalLM( 
            GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m") 
        )
    # TODO: based
    else:
        raise NotImplementedError
    hps["max_steps"]=200_000 # DELETE
    model = model_cls(model, **model_kwargs)
    
    print(model)
    print(task)

    training_args = TrainingArguments(
        output_dir="trash",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_safetensors=False, # workaround for Runtime: tensors with shared memory error
        report_to="wandb",
        disable_tqdm=disable_tqdm,
        **hps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # callbacks=[WandbAlphasCallback(freq=1)],
        **trainer_kwargs,
    )
    trainer.train()
    
    print("Took {} sec".format(time.time()-start))

if __name__ == "__main__":
    fire.Fire(main)