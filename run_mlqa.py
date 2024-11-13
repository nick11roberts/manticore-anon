import sys
import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
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
from data.natural_language.natural_instructions import get_NI_dataset
from models.mamba_custom import (
    MambaConfigHF,
    MambaForCausalLM,
    MambaForTokenClassification,
    MambaForSequenceClassification,
)
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
        model_type: str = "manticore",
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
        projector_type=projector_type,
        projector_gpt=projector_gpt,
        vocab_size=trainer_kwargs["tokenizer"].vocab_size,
        dash=dash,
        gaea=gaea,
        retraining=retraining,
    )

    mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    # pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    # gptneo =

    if model_type == "manticore":
        models = [mamba, pythia]
        # NOTE replace with pretrained gptneo and mamba models
        model_kwargs.pop("gptneo", None)
        model_kwargs.pop("pythia", None)
        model_kwargs.pop("mamba", None)

        model = model_clses["manticore"](
            config=config,
            models=models,
            pretrain=pretrain,
            **model_kwargs,
        ).to(device=device)

        if "num_classes" in model_kwargs.keys():
            config.num_labels = model_kwargs.pop("num_classes", None)
        # model = mamba

        if load_projectors:
            model.load_projectors("manticore-780m") # TODO add path to projector checkpoint
    elif model_type == 'mamba':
        config = MambaConfigHF(model_kwargs["mamba"].config)
        if "num_classes" in model_kwargs.keys():
            config.num_labels = model_kwargs.pop("num_classes", None)
        model = model_clses["mamba"](config=config).to(device=device)
    elif model_type == 'pythia':
        model = pythia
    elif model_type == 'pythia160':
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    elif model_type == 'pythia160d':
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped")
    elif model_type == 'pythia410d':
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")
    elif model_type == 'pythia410c':
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-capitals")
    elif model_type == 'gptneo':
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

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
    per_device_batch_size=2,
    gradient_accumulation_steps=4,
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
        model_type=model_type,
    )

    print(model)
    print(task)

    # train_dataset = train_dataset.select(range(len(train_dataset) // data_frac))

    if model_type == "manticore":
        hps.pop("per_device_train_batch_size", None)  # Default 8
        hps["per_device_train_batch_size"] = per_device_batch_size
        hps.pop("gradient_accumulation_steps", None)
        hps["gradient_accumulation_steps"] = gradient_accumulation_steps
        model.enable_alphas()  # ✅ enable alphas
        model.enable_finetuning()  # ✅ enable fine-tuning
        model.enable_projectors()  # ✅ enable projector learning
    if gaea: opt_type_str = "_gaea"
    elif dash: opt_type_str = "_dash"
    else: opt_type_str = ""
    
    run_name = f"mlqa_{task}{opt_type_str}"
    output_dir = os.path.join("/data/manticore", run_name) 
    training_args = TrainingArguments(
        output_dir= output_dir,
        overwrite_output_dir=True,
        run_name = run_name,
        # evaluation_strategy="steps",
        # eval_steps=100,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        save_total_limit=1,
        report_to="wandb",
        disable_tqdm=disable_tqdm,
        logging_steps=10,
        num_train_epochs=15,
        **hps,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # 如果验证集损失 3 个评估周期没有改善则停止训练
        early_stopping_threshold=0.0  # 改善必须超过阈值 0.0
    )
    # if gaea:
    #     trainer = MixedOptTrainer(
    #         arch_lr = 1e-2,
    #         alternating=False, ## alternating doesnt work with dataset 
    #         steps_per_opt=None,
    #         ##
    #         model=model,
    #         # model_init=model_init, #EDIT
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         callbacks=[WandbAlphasCallback(freq=1), early_stopping_callback],
    #         **trainer_kwargs,
    #     )
    # else:
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         callbacks=[WandbAlphasCallback(freq=1), early_stopping_callback],
    #         **trainer_kwargs,
    #     )
    trainer = MixedOptTrainer(
        arch_lr = 1e-2 if gaea else 1e-3,
        gaea=gaea,
        alternating=False, ## alternating doesnt work with dataset 
        steps_per_opt=None,
        ##
        model=model,
        # model_init=model_init, #EDIT
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbAlphasCallback(freq=1), early_stopping_callback],
        **trainer_kwargs,
    )

    # from transformers import get_linear_schedule_with_warmup
    # from run_2_optimizer import MixedOptimizer, MixedScheduler
    #
    # batch_size = 8
    # epochs = 1
    # num_steps = ((len(train_dataset)) // batch_size) * epochs
    #
    # optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    # p_arch = [p for name, p in model.named_parameters() if "alphas" in name]
    # p_model = [p for name, p in model.named_parameters() if "alphas" not in name]
    #
    # optim_model = optim_cls(p_model, **optim_kwargs)
    # lr = optim_kwargs.pop("lr", None)
    # # print(lr)
    # # quit()
    # optim_arch = optim_cls(p_arch, lr=arch_lr, **optim_kwargs)
    #
    # # sched_model = get_linear_schedule_with_warmup(
    # #     optim_model, num_warmup_steps=0, num_training_steps=num_steps
    # # )
    # # sched_arch = get_linear_schedule_with_warmup(
    # #     optim_arch, num_warmup_steps=0, num_training_steps=num_steps
    # # )
    #
    # optimizer = MixedOptimizer([optim_model, optim_arch], alternating=alternating)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=num_steps
    # )
    # # scheduler = MixedScheduler(optimizer, [sched_model, sched_arch])
    # optims = (optimizer, scheduler)
    #
    # # print(optimizer)
    # # quit()
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     callbacks=[WandbAlphasCallback(freq=1)],
    #     optimizers=optims,
    #     **trainer_kwargs,
    # )
    trainer.train()

    if model_type == "manticore" and alpha_mamba is None and retraining:
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
            gaea=gaea,
            retraining=True,
        )

        model_retrain.load_alphas(model)
        model_retrain.disable_alphas()
        del model

        run_name = "RETRAIN_" + training_args.run_name
        output_dir = os.path.join("/data/manticore/retrain", training_args.run_name) 
        training_args.run_name = run_name
        training_args.output_dir = output_dir
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
