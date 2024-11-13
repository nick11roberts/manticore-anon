import sys
import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import TrainingArguments, Trainer
from mamba_ssm import MambaLMHeadModel
from models.manticore_2 import (  # TODO
    ManticoreConfig,
    ManticoreForTokenClassification,
    ManticoreForLEGO,
    ManticoreForICLRegression,
    ManticoreForCausalLM,
)
from models.mamba_custom import MambaForCausalLM, MambaConfigHF
from callbacks.wandb import WandbAlphasCallback
from data.lego.lego_data import lego_dataset
from data.icl_regression.data import get_regression_dataset, RegressionIterDataset
from data.natural_language.eli5 import get_eli5_dataset
from data.natural_language.ptb import get_ptb_dataset
from data.natural_language.alpaca import get_alpaca_dataset
from nas import MixedOptimizer, MixedDataset
from data.automata.automata import AutomatonDataset

from run_configs_2 import get_experiment_config  # TODO

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"


def main(
    task: str = "fineweb",
    model_type: str = "manticore",
    d_model=768,
    n_mixtures=12,
    use_projectors=True,
    pretrain=False,
    epochs=10,
    device="cuda",
    disable_tqdm=False,
    alpha_mamba=None,
    projector_type="default",
    projector_gpt=True,
    per_device_batch_size=8,
    gradient_accumulation_steps=1,
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
    )

    if model_type == "manticore":
        config = ManticoreConfig(
            d_model=d_model,
            n_mixtures=n_mixtures,
            use_projectors=use_projectors,
            alpha_mamba=alpha_mamba,
            projector_type=projector_type,
            projector_gpt=projector_gpt,
        )

        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        #
        # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        # pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
        #
        # "Medium-scale"
        # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-1.3b")
        # pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped")

        # models = [mamba, pythia]
        models = [gptneo, mamba2]

        # NOTE replace with pretrained gptneo and mamba models
        model_kwargs.pop("mamba", None)
        model_kwargs.pop("gptneo", None)

        model = model_clses["manticore"](
            config=config,
            # mamba=mamba,
            # gptneo=gptneo,
            models=models,
            pretrain=pretrain,
            **model_kwargs,
        ).to(device=device)
    else:
        raise NotImplementedError

    print(config)

    print(model)
    print(model.num_parameters())
    print(task)

    model.enable_alphas()  # ✅ enable alphas
    model.disable_finetuning()  # ❌ disable fine-tuning
    model.enable_projectors()  # ✅ enable projector learning
    model.enable_embeddings()  # ✅ enable embedding learning

    hps.pop("per_device_train_batch_size", None)  # Default 8
    hps["per_device_train_batch_size"] = per_device_batch_size
    hps.pop("per_device_eval_batch_size", None)  # Default 8
    hps["per_device_eval_batch_size"] = per_device_batch_size
    hps.pop("gradient_accumulation_steps", None)
    hps["gradient_accumulation_steps"] = gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir="projector_checkpoints_fineweb_3b",
        eval_strategy="no",
        save_strategy="steps",
        # save_steps=1000,
        save_steps=100,
        save_safetensors=False,
        report_to="wandb",
        disable_tqdm=disable_tqdm,
        ignore_data_skip=True,
        **hps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[WandbAlphasCallback(freq=1)],
        **trainer_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
