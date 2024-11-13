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
from data.natural_language.mlqa import get_mlqa_dataset

from nas import (
    MixedOptimizer, MixedDataset, 
    create_manticore_mixed_optimizer, MixedOptTrainer
)
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
    projector_type="default",
    projector_gpt=True,
    load_projectors=True,
    dash=False,
    gaea=False,
    retraining=False,
):
    assert not (dash and gaea), "Cannot specify both a GAEA and DASH style update"
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
        projector_type=projector_type,
        projector_gpt=projector_gpt,
        vocab_size=trainer_kwargs["tokenizer"].vocab_size,
        dash=dash,
        gaea=gaea,
        retraining=retraining,
    )

    mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    # mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # mamba2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

    models = [mamba, pythia]
    # NOTE replace with pretrained gptneo and mamba models
    model_kwargs.pop("mamba", None)
    model_kwargs.pop("pythia", None)
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
        model.load_projectors("manticore-780m") # TODO add path to projector checkpoint
    print(train_dataset, eval_dataset)
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
    projector_type="default",
    projector_gpt=True,
    data_frac=1,
    arch_lr=0.01,
    alternating=False,
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    retraining=True,
    dash=False,
    gaea=False,
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
        projector_type,
        projector_gpt,
        load_projectors,
        dash=dash,
        gaea=gaea,
        retraining=False,
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
    if gaea:
        trainer = MixedOptTrainer(
            arch_lr = 1e-1,
            alternating=False, ## alternating doesnt work with dataset 
            steps_per_opt=None,
            ##
            model=model,
            # model_init=model_init, #EDIT
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[WandbAlphasCallback(freq=1)],
            **trainer_kwargs,
        )
    else:
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

    optim_model = optim_cls(p_model, **optim_kwargs)
    lr = optim_kwargs.pop("lr", None)
    # print(lr)
    # quit()
    optim_arch = optim_cls(p_arch, lr=arch_lr, **optim_kwargs)

    # sched_model = get_linear_schedule_with_warmup(
    #     optim_model, num_warmup_steps=0, num_training_steps=num_steps
    # )
    # sched_arch = get_linear_schedule_with_warmup(
    #     optim_arch, num_warmup_steps=0, num_training_steps=num_steps
    # )

    optimizer = MixedOptimizer([optim_model, optim_arch], alternating=alternating)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )
    # scheduler = MixedScheduler(optimizer, [sched_model, sched_arch])
    optims = (optimizer, scheduler)

    # print(optimizer)
    # quit()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbAlphasCallback(freq=1)],
        optimizers=optims,
        **trainer_kwargs,
    )
    trainer.train()

    if alpha_mamba is None and retraining:
        print("---------RUNNING POST-SEARCH RETRAINING!!!---------")

        # NOTE POST SEARCH RETRAINING
        model_retrain, _, _, _, _ = get_model(
            task,
            d_model,
            n_mixtures,
            use_projectors,
            pretrain,
            epochs,
            device,
            alpha_mamba,
            projector_type,
            projector_gpt,
            load_projectors,
            dash=False,
            gaea=False,
            retraining=True,
        )

        model_retrain.load_alphas(model)
        model_retrain.disable_alphas()

        trainer = Trainer(
            model=model_retrain,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[WandbAlphasCallback(freq=1)],
            # optimizers=optims,
            **trainer_kwargs,
        )
        trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
