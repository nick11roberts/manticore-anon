import sys
import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

from datasets import load_dataset

from mamba_ssm import MambaLMHeadModel

# from based.models.gpt import GPTLMHeadModel
from models.manticore import (
    ManticoreConfig,
    ManticoreForTokenClassification,
    ManticoreForLEGO,
    ManticoreForICLRegression,
    ManticoreForCausalLM,
)

from callbacks.wandb import WandbAlphasCallback
import wandb

from data.lego.lego_data import lego_dataset
from data.icl_regression.data import get_regression_dataset, RegressionIterDataset

from nas import MixedOptimizer, MixedDataset
import copy

device = "cuda"

"""TODOs:
-Add in Automaton dataset and remove NotImplementedError raises
-Other eli5 subtasks?
-Add eval metrics for each task; compute_metric in Trainer args?
"""


class AutomatonDataset:
    def __init__(self, name: str, ntrain: int):
        # ntrain = 600_000  # 2048  # 600_000 # 5_000_000
        length = 100

        # TODO subclass
        if name == "abab":
            input_vocab_size = 2
            output_vocab_size = 5
        elif name == "add":
            raise NotImplementedError
        elif name == "alternating":
            raise NotImplementedError
        elif name == "cyclic":
            input_vocab_size = 2
            output_vocab_size = 5
        elif name == "dihedral":
            raise NotImplementedError
        elif name == "flipflop":
            input_vocab_size = 3
            output_vocab_size = 3
        elif name == "gridworld":
            input_vocab_size = 2
            output_vocab_size = 9
        elif name == "parity":
            input_vocab_size = 2
            output_vocab_size = 2
        elif name == "quaternion":
            input_vocab_size = 4
            output_vocab_size = 8
        elif name == "symmetric":
            raise NotImplementedError
        elif name == "permutation_reset":
            raise NotImplementedError
        else:
            raise NotImplementedError

        datapath = "data/synthseq/automata/automata.py"

        config_train = {"size": ntrain, "name": name, "length": length}
        dataset_train = load_dataset(
            datapath,
            config=config_train,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # config_train.cleanup_cache_files()

        config_val = {"size": 2048, "name": name, "length": length}
        dataset_val = load_dataset(
            datapath,
            config=config_val,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # dataset_val.cleanup_cache_files()

        config_test = {"size": 2048, "name": name, "length": length}
        dataset_test = load_dataset(
            datapath,
            config=config_test,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # dataset_test.cleanup_cache_files()

        self.length = length
        self.ntrain = ntrain
        self.name = name
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test


def get_experiment_config(task_type, task_name, ntrain):
    """######### Create datasets"""
    if task_type == "automata":
        # raise NotImplementedError
        dataset = AutomatonDataset(task_name, ntrain)
        eval_dataset = dataset.dataset_val["train"]
        model_cls = ManticoreForTokenClassification
        vocab_size = dataset.output_vocab_size
    elif task_type == "regression":
        dataset = get_regression_dataset(
            task_name, n_dims=20, n_dims_truncated=5, batch_size=64, n_points=41
        )
        assert dataset.curriculum is not None
        eval_dataset = get_regression_dataset(
            task_name,
            n_dims=20,
            n_dims_truncated=20,
            batch_size=64,
            n_points=(
                101 if (task_name in ["relu_2nn_regression", "decision_tree"]) else 41
            ),
            n_samples=ntrain // 5,
        )
        model_cls = ManticoreForICLRegression
    elif task_type == "lego":
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # from MAMBA; maybe make a smaller vocab custom one?
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer
        dataset = lego_dataset(
            tokenizer,
            n_var=12,
            n_samples=ntrain,
            batch_size=100,
            random_order=True,
            append_vars_sep_token="<|endoftext|>",
        )
        eval_dataset = lego_dataset(
            tokenizer,
            n_var=12,
            n_samples=ntrain,
            batch_size=100,
            random_order=True,
            append_vars_sep_token="<|endoftext|>",
        )
        model_cls = ManticoreForLEGO
    elif "eli5" in task_type:
        eli5 = load_dataset(
            "eli5_category", split="train"
        )  # TODO: there are other eli5 task_names? eli5 is the task type?
        eli5 = eli5.train_test_split(test_size=0.05, seed=1111)  # TODO maybe change
        eli5 = eli5.flatten()
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer
        """Stuff for eli5"""

        def preprocess_function(examples):
            return tokenizer([" ".join(x) for x in examples["answers.text"]])

        tokenized_eli5 = eli5.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=eli5["train"].column_names,
        )
        block_size = 128

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of block_size.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        dataset = lm_dataset["train"]
        eval_dataset = lm_dataset["test"]

        model_cls = ManticoreForCausalLM
    else:
        raise NotImplementedError
    if task_type == "automata":
        # raise NotImplementedError
        dataset = dataset.dataset_train["train"]

    """####### Hyperparameters; kwargs for TrainerArguments and Trainer"""
    if task_type == "automata":
        # raise NotImplementedError
        model_kwargs = {"num_classes": vocab_size}
        hps = {
            "learning_rate": 3e-5,
            "weight_decay": 1e-4,
            "num_train_epochs": 1,  ##
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
        }
        trainer_kwargs = {}
    elif task_type == "regression":
        assert isinstance(dataset, RegressionIterDataset)
        model_kwargs = {
            "n_regr_dims": dataset.n_dims,
            "train_metric_fn": dataset.task_sampler().get_training_metric(),
        }
        hps = {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,  # ?
            # "num_train_epochs": 1,
            "lr_scheduler_type": "constant",
            "max_steps": 2,  # 500_000 from ICL transformer paper, but convergence with curriculum happens faster?
            "per_device_train_batch_size": dataset.batch_size,
            "per_device_eval_batch_size": dataset.batch_size,
        }
        trainer_kwargs = {}
    elif task_type == "lego":
        model_kwargs = {}
        grad_accum_steps = 4
        hps = {
            "learning_rate": 5e-5,
            "weight_decay": 0.0,  # ?
            "num_train_epochs": 60,  # 200
            "gradient_accumulation_steps": grad_accum_steps,
            "per_device_train_batch_size": 1000
            // grad_accum_steps,  # >=500 batch size per grad update recommended by lego paper, but will be problematic for memory
            "per_device_eval_batch_size": 200,
        }
        trainer_kwargs = {}
    elif task_type == "eli5":
        model_kwargs = {}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "projectors_eli5":
        model_kwargs = {}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 20,  # 200
            # "per_device_train_batch_size":500, # TODO?
            # "per_device_eval_batch_size":500
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "projectors_eli5":
        model_kwargs = {}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 20,  # 200
            # "per_device_train_batch_size":500, # TODO?
            # "per_device_eval_batch_size":500
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    else:
        raise NotImplementedError

    return model_cls, model_kwargs, hps, trainer_kwargs, dataset, eval_dataset


def main(
    task: str = "regression:linear_regression",
    model_type: str = "manticore",
    d_model=768,
    n_mixtures=4,
    use_projectors=True,
    fixed_alphas=False,
    alphas_descent_type="darts",
    pretrain=False,
    load_projectors=True,
    device="cuda",
    log_wandb=True,
    disable_tqdm=False,
):

    task_split = task.split(":")
    if len(task_split) == 1:
        task_split.append(None)
    task_type, task_name = task_split
    print(task_type, task_name)

    if model_type == "manticore":
        config = ManticoreConfig(
            d_model=d_model,
            n_mixtures=n_mixtures,
            fixed_alphas=fixed_alphas,
            use_projectors=use_projectors,
            alphas_descent_type=alphas_descent_type
        )

        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # based = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m")

        model_cls, model_kwargs, hps, trainer_kwargs, train_dataset, eval_dataset = (
            get_experiment_config(task_type, task_name, ntrain=50_000)
        )

        model = model_cls(
            config=config,
            mamba=mamba,
            gptneo=gptneo,
            # based=based
            pretrain=pretrain,
            **model_kwargs,
        ).to(device=device)
        print(config)
    elif model_type == "mamba":
        pass
    elif model_type == "gptneo":
        pass
    else:
        raise NotImplementedError

    print(model)
    print(task)

    model.disable_alphas() # ❌ disable alphas
    model.disable_finetuning() # ❌ disable fine-tuning
    model.enable_projectors() # ✅ enable projector learning
    training_args = TrainingArguments(
        output_dir="trash" if "projectors_" not in task else "projector_checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        eval_dataset=eval_dataset,
        callbacks=[WandbAlphasCallback(freq=1)],
        **trainer_kwargs,
    )
    if load_projectors:
        projector_checkpoint = None # NOTE add projector checkpoint path
        model = ManticoreForCausalLM.from_pretrained(
            projector_checkpoint, mamba=mamba, gptneo=gptneo, 
        )
        print(f"Projectors loaded from {projector_checkpoint}")
    else:
        trainer.train()

    model.enable_alphas() # ✅ disable alphas
    model.disable_finetuning() # ❌ disable fine-tuning
    model.disable_projectors( )# ❌ enable projector learning
    training_args = TrainingArguments(
        output_dir="trash",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=5000,
        save_safetensors=False, 
        report_to="wandb",
        disable_tqdm=disable_tqdm,
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
