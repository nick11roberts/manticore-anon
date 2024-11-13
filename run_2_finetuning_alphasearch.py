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


def get_model(
    task: str = "alpaca",
    d_model=768,
    n_mixtures=4,
    use_projectors=True,
    pretrain=False,
    epochs=10,
    device="cuda",
    alpha_mamba=None,
    alpha_mamba2=None,
    projector_type="default",
    projector_gpt=True,
    load_projectors=True,
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

    config = ManticoreConfig(
        d_model=d_model,
        n_mixtures=n_mixtures,
        use_projectors=use_projectors,
        alpha_mamba=alpha_mamba,
        alpha_mamba2=alpha_mamba2,
        projector_type=projector_type,
        projector_gpt=projector_gpt,
        vocab_size=trainer_kwargs["tokenizer"].vocab_size,
    )

    # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    # mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    # mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

    models = [mamba, pythia]

    # NOTE replace with pretrained gptneo and mamba models
    model_kwargs.pop("mamba", None)
    model_kwargs.pop("gptneo", None)

    model = model_clses["manticore"](
        config=config,
        models=models,
        pretrain=pretrain,
        **model_kwargs,
    ).to(device=device)
    # print(config)

    if load_projectors:
        # model.load_projectors("./projector_checkpoints_fineweb/checkpoint-63000")
        model.load_projectors("./projector_checkpoints_fineweb_large/checkpoint-63000")

    return model, hps, trainer_kwargs, train_dataset, eval_dataset


def main(
    task: str = "alpaca",
    model_type: str = "manticore",
    d_model=768,
    n_mixtures=4,
    use_projectors=True,
    pretrain=False,
    load_projectors=True,
    epochs=10,
    device="cuda",
    disable_tqdm=False,
    alpha_mamba=None,
    alpha_mamba2=None,
    projector_type="default",
    projector_gpt=True,
    data_frac=1,
    arch_lr=0.01,
    alternating=False,
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    retraining=True,
):

    model, hps, trainer_kwargs, train_dataset, eval_dataset = get_model(
        task,
        d_model,
        n_mixtures,
        use_projectors,
        pretrain,
        epochs,
        device,
        alpha_mamba,
        alpha_mamba2,
        projector_type,
        projector_gpt,
        load_projectors,
    )

    print(model)
    print(task)

    train_dataset = train_dataset.select(range(len(train_dataset) // data_frac))

    hps.pop("per_device_train_batch_size", None)  # Default 8
    hps["per_device_train_batch_size"] = per_device_batch_size
    hps.pop("gradient_accumulation_steps", None)
    hps["gradient_accumulation_steps"] = gradient_accumulation_steps

    model.enable_alphas()  # ✅ enable alphas
    model.enable_finetuning()  # ✅ enable fine-tuning
    model.enable_projectors()  # ✅ enable projector learning
    training_args = TrainingArguments(
        output_dir="trash",
        # evaluation_strategy="steps",
        # eval_steps=100,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_safetensors=False,
        report_to="wandb",
        disable_tqdm=disable_tqdm,
        logging_steps=10,
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
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
