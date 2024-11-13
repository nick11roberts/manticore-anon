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
from models.manticore_2 import (
    ManticoreConfig,
    ManticoreForTokenClassification,
    ManticoreForLEGO,
    ManticoreForICLRegression,
    ManticoreForCausalLM,## only this one actually gets used
    ManticoreForSequenceClassification,
)
from models.mamba_custom import (
    MambaConfigHF,
    MambaForCausalLM,
    MambaForTokenClassification,
    MambaForSequenceClassification,
)
from callbacks.wandb import WandbAlphasCallback
from data.lego.lego_data import lego_dataset
from data.icl_regression.data import get_regression_dataset, RegressionIterDataset
from data.natural_language.eli5 import get_eli5_dataset
from data.natural_language.ptb import get_ptb_dataset
from data.natural_language.mix import get_mix_dataset
from data.natural_language.alpaca import get_alpaca_dataset
from data.natural_language.fineweb import get_fineweb_dataset
from data.natural_language.gsm8k import get_gsm8k_dataset
from data.natural_language.mlqa import get_mlqa_dataset
from data.natural_language.natural_instructions import get_NI_dataset
from data.natural_language.openorcha import get_openorcha_dataset
from data.natural_language.squad import get_squad_dataset

# from data.prepare_lra import get_lra_dataset
# from data.mad.get_mad import get_mad
from nas import MixedOptimizer, MixedDataset
from data.automata.automata import AutomatonDataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"


def get_experiment_config(task_type, task_name, ntrain, d_model, n_mixtures, epochs):
    """####### Hyperparameters, datasets, kwargs for TrainerArguments and Trainer"""
    # if task_type == "mad":
    #     dataset, eval_dataset, mad_config = get_mad(task_name=task_name)

    #     model_clses = {
    #         "manticore": ManticoreForTokenClassification,
    #         "mamba": MambaForTokenClassification,
    #         "gptneo": GPTNeoForTokenClassification,
    #     }

    #     mamba_config = MambaConfig(
    #         d_model=d_model,
    #         n_layer=2 * n_mixtures,
    #         vocab_size=mad_config.vocab_size,
    #     )
    #     mamba = MambaLMHeadModel(mamba_config)
    #     print(mamba)
    #     num_mamba_params = sum(p.numel() for p in mamba.parameters())
    #     print("num_mamba_params", num_mamba_params)

    #     gptneo_config = GPTNeoConfig(
    #         vocab_size=mad_config.vocab_size,
    #         hidden_size=d_model,
    #         num_layers=n_mixtures,
    #         attention_types=[[["global"], n_mixtures]],
    #         num_heads=16,
    #         intermediate_size=4 * d_model,
    #     )
    #     gptneo = GPTNeoForCausalLM(gptneo_config)
    #     print(gptneo)
    #     num_gptneo_params = sum(p.numel() for p in gptneo.parameters())
    #     print("num_gptneo_params", num_gptneo_params)

    #     model_kwargs = {
    #         "mamba": mamba,
    #         "gptneo": gptneo,
    #         # "ignore_index": mad_config.target_ignore_index,
    #         "num_classes": mad_config.vocab_size,
    #     }
    #     hps = {
    #         "learning_rate": 5e-4,
    #         "weight_decay": 0.00,
    #         "num_train_epochs": epochs,  # ?
    #         "per_device_train_batch_size": 128,
    #     }

    #     # Setup evaluation
    #     import evaluate

    #     metric = evaluate.load("accuracy")

    #     def compute_metrics(eval_pred):
    #         logits, labels = eval_pred
    #         predictions = np.argmax(logits, axis=-1)
    #         labels = labels.reshape(-1)
    #         predictions = predictions.reshape(-1)
    #         inds = np.argwhere(labels != -100)[:, 0]

    #         return metric.compute(
    #             predictions=predictions[inds],
    #             references=labels[inds],
    #         )

    #     trainer_kwargs = {
    #         "compute_metrics": compute_metrics,
    #     }

    if task_type == "automata":
        dataset = AutomatonDataset(task_name, ntrain)
        eval_dataset = dataset.dataset_val["train"]
        model_clses = {
            "manticore": ManticoreForTokenClassification,
            "mamba": MambaForTokenClassification,
            "gptneo": GPTNeoForTokenClassification,
        }
        vocab_size = dataset.output_vocab_size
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {"mamba": mamba, "gptneo": gptneo, "num_classes": vocab_size}
        hps = {
            "learning_rate": 3e-5,
            "weight_decay": 1e-4,
            "num_train_epochs": 1,  ##
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
        }
        trainer_kwargs = {}
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
        model_clses = {
            "manticore": ManticoreForICLRegression,
            "mamba": None,  # TODO need to implement these for GPTNeo and Mamba
            "gptneo": None,  # TODO need to implement these for GPTNeo and Mamba
        }
        assert isinstance(dataset, RegressionIterDataset)
        # mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # # TODO edit config to get model of appropriate size for the task
        # mamba = MambaLMHeadModel(mamba.config)
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # # TODO edit config to get model of appropriate size for the task
        # gptneo = GPTNeoForCausalLM(gptneo.config)
        mamba_config = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-130m"
        ).config
        mamba_config.vocab_size = 1  # embedder will be ignored anyway
        mamba_config.d_model = 256
        mamba_config.n_layer = 12  # ~5M
        mamba = MambaLMHeadModel(mamba_config)

        gptneo_config = GPTNeoConfig(
            num_layers=12,
            hidden_size=256,
            attention_types=[[["global", "local"], 6]],
            vocab_size=1,  # embedder will be ignored anyway
        )  # ~10M
        gptneo = GPTNeoForCausalLM(gptneo_config)

        model_kwargs = {
            "mamba": mamba,
            "gptneo": gptneo,
            "n_regr_dims": dataset.n_dims,
            "train_metric_fn": dataset.task_sampler().get_training_metric(),
        }
        hps = {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,  # ?
            # "num_train_epochs": 1,
            "lr_scheduler_type": "constant",
            "max_steps": 100_000,  # 500_000 from ICL transformer paper, but convergence with curriculum happens faster?
            "per_device_train_batch_size": dataset.batch_size,
            "per_device_eval_batch_size": dataset.batch_size,
        }
        trainer_kwargs = {}
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
        model_clses = {
            "manticore": ManticoreForLEGO,
            "mamba": None,  # TODO need to implement these for GPTNeo and Mamba
            "gptneo": None,  # TODO need to implement these for GPTNeo and Mamba
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {"mamba": mamba, "gptneo": gptneo}
        grad_accum_steps = 4
        hps = {
            "learning_rate": 5e-4,
            "weight_decay": 0.0,  # ?
            "num_train_epochs": 60,  # 200
            "gradient_accumulation_steps": grad_accum_steps,
            "per_device_train_batch_size": 1000
            // grad_accum_steps,  # >=500 batch size per grad update recommended by lego paper, but will be problematic for memory
            "per_device_eval_batch_size": 200,
        }
        trainer_kwargs = {}
    elif task_type == "eli5":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


        dataset, eval_dataset, data_collator = get_eli5_dataset(tokenizer)
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {
            "mamba": mamba,
            "gptneo": gptneo,
        }
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "mix":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # TODO figure out which tokenizer to use and when

        dataset, eval_dataset, data_collator = get_mix_dataset(
            tokenizer, slice=task_name
        )
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {"mamba": mamba, "gptneo": gptneo}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "ptb":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # TODO figure out which tokenizer to use and when

        dataset, eval_dataset, data_collator = get_ptb_dataset(tokenizer)
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {"mamba": mamba, "gptneo": gptneo}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "fineweb":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_fineweb_dataset(tokenizer)
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {"mamba": mamba, "gptneo": gptneo}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
            "per_device_train_batch_size": 8,
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}

    elif task_type == "alpaca":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_alpaca_dataset(tokenizer)
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}

    elif task_type == "gsm8k":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_gsm8k_dataset(tokenizer)
        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
        # TODO edit config to get model of appropriate size for the task
        mamba = MambaLMHeadModel(mamba.config)
        gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
        # TODO edit config to get model of appropriate size for the task
        gptneo = GPTNeoForCausalLM(gptneo.config)
        model_kwargs = {}
        hps = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,  # ?
            "num_train_epochs": 1,  # 200
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "mlqa":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_mlqa_dataset(tokenizer, task_name)
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

        pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        model_kwargs = {"mamba": mamba, "pythia": pythia}
        hps = {
            # "learning_rate": 2e-4,
            # "weight_decay": 0.01,
            # "num_train_epochs": 1,
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "natural_instructions":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_NI_dataset(tokenizer, task_name)
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

        pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        model_kwargs = {"mamba": mamba, "pythia": pythia}
        hps = {
            # "learning_rate": 2e-4,
            # "weight_decay": 0.01,
            # "num_train_epochs": 1,
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}
    elif task_type == "openorcha":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_openorcha_dataset(tokenizer)
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

        pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        model_kwargs = {"mamba": mamba, "pythia": pythia}
        hps = {
            # "learning_rate": 2e-4,
            # "weight_decay": 0.01,
            # "num_train_epochs": 1,
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}

    elif task_type == "squad":
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B"
        )  # GPT-Neo tokenizer

        dataset, eval_dataset, data_collator = get_squad_dataset(tokenizer, task_name)
        mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
        # gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

        pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

        model_clses = {
            "manticore": ManticoreForCausalLM,
            "mamba": MambaForCausalLM,
            "gptneo": GPTNeoForCausalLM,
        }
        model_kwargs = {"mamba": mamba, "pythia": pythia}
        hps = {
            # "learning_rate": 2e-4,
            # "weight_decay": 0.01,
            # "num_train_epochs": 1,
        }
        trainer_kwargs = {"data_collator": data_collator, "tokenizer": tokenizer}

    # elif task_type == "lra":
    #     (
    #         dataset,
    #         eval_dataset,
    #         num_class,
    #         sequence_length,
    #         hps,
    #         gptneo_config,
    #         mamba_config,
    #         pad_token_id,
    #     ) = get_lra_dataset(task_name)

    #     layers_count = gptneo_config.num_layers
    #     dimensions = gptneo_config.hidden_size

    #     # Setup evaluation
    #     import evaluate

    #     metric = evaluate.load("accuracy")

    #     def compute_metrics(eval_pred):
    #         logits, labels = eval_pred
    #         predictions = np.argmax(logits, axis=-1)
    #         labels = labels.reshape(-1)
    #         predictions = predictions.reshape(-1)
    #         inds = np.argwhere(labels != -100)[:, 0]

    #         return metric.compute(
    #             predictions=predictions[inds],
    #             references=labels[inds],
    #         )

    #     trainer_kwargs = {
    #         "compute_metrics": compute_metrics,
    #     }

    #     if pad_token_id is not None:
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             "EleutherAI/gpt-neo-1.3B"
    #         )  # GPT-Neo tokenizer
    #         tokenizer.pad_token_id = 0
    #         trainer_kwargs["tokenizer"] = tokenizer
    #     print(trainer_kwargs)

    #     model_clses = {
    #         "manticore": ManticoreForSequenceClassification,
    #         "mamba": MambaForSequenceClassification,
    #         "gptneo": GPTNeoForSequenceClassification,
    #     }
    #     mamba = MambaLMHeadModel(mamba_config)
    #     gptneo = GPTNeoForCausalLM(gptneo_config)
    #     model_kwargs = {"mamba": mamba, "gptneo": gptneo, "num_classes": num_class}

    else:
        raise NotImplementedError

    if "projectors" in task_type:
        hps["num_train_epochs"] = 10

    return (
        model_clses,
        model_kwargs,
        hps,
        trainer_kwargs,
        dataset,
        eval_dataset,
    )
