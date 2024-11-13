import json
import os
import torch
import numpy as np
from PIL import Image, ImageOps

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoConfig
from mamba_ssm import MambaLMHeadModel
from transformers import GPTNeoForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
import sys
sys.path.append("../")
from long_range_arena.dataloaders import lra


def lra_text_classification(dataset_name: str):
    if dataset_name != "imdb":
        raise ValueError("Invalid dataset name")
    data = lra.IMDB(dataset_name, l_max=2048)
    data.setup()
    dataset, tokenizer, vocab = data.process_dataset()
    return dataset['train'], dataset['test'], 2, 2048


def lra_pixel_image_classification(dataset_name: str):
    if dataset_name == "mnist":
        dataset_ = load_dataset("mnist")
        input_data = [np.array(i['image']).flatten() for i in dataset_['train']]
        label = [i['label'] for i in dataset_['train']]
        train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
        input_data_test = [np.array(i['image']).flatten() for i in dataset_['test']]
        label_test = [i['label'] for i in dataset_['test']]
        test_dataset = Dataset.from_dict({"input_ids": input_data_test, "labels": label_test})
    elif dataset_name == "cifar10":
        dataset_ = load_dataset("cifar10")
        input_data = [np.array(ImageOps.grayscale(i["img"])).flatten() for i in dataset_['train']]
        label = [i['label'] for i in dataset_['train']]
        train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
        input_data_test = [np.array(ImageOps.grayscale(i["img"])).flatten() for i in dataset_['test']]
        label_test = [i['label'] for i in dataset_['test']]
        test_dataset = Dataset.from_dict({"input_ids": input_data_test, "labels": label_test})
    else:
        raise ValueError("Invalid dataset name")
    num_of_labels = len(set(train_dataset["labels"]))
    sequence_length = len(train_dataset["input_ids"][0])
    return train_dataset, test_dataset, num_of_labels, sequence_length


# split: dataset["train"], dataset["val"], dataset["test"]
def prepare_listops_data(data_dir: str):
    data = lra.ListOps('listops', data_dir=data_dir)
    data.setup()
    dataset, tokenizer, vocab = data.process_dataset()

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_dataset = train_dataset.rename_column("Target", "label")
    test_dataset = test_dataset.rename_column("Target", "label")
    num_class = len(set(train_dataset["label"]))
    sequence_length = 2000
    print(data.vocab_size)
    return train_dataset, test_dataset, num_class, sequence_length


# split: ds.dataset_train, ds.dataset_val, ds.dataset_test
def prepare_pathfinder_data(data_dir: str):
    dataset = lra.PathFinder('pathfinder', data_dir=data_dir, tokenize=True)
    dataset.setup()

    input_data = [i[0] for i in dataset.dataset_train]
    label = [i[1] for i in dataset.dataset_train]
    train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
    input_data_test = [i[0] for i in dataset.dataset_test]
    label_test = [i[1] for i in dataset.dataset_test]
    test_dataset = Dataset.from_dict({"input_ids": input_data_test, "labels": label_test})
    num_class = 2
    sequence_length = len(train_dataset["input_ids"][0])
    return train_dataset, test_dataset, num_class, sequence_length


def get_lra_dataset(dataset_name: str):
    pad_token_id = None
    mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

    if dataset_name == "listops":
        train_dataset, test_dataset, num_class, sequence_length = prepare_listops_data(
            "long_range_arena/datasets/listops-1000")
        hps = {
            "max_steps": 5000
            }

        gptneo.config.num_heads = 8
        gptneo.config.num_layers = 6
        gptneo.config.hidden_size = 512
        gptneo.config.intermediate_size = 2048
        gptneo.config.attention_types = [[['global', 'local'], 3]]
        gptneo.config.attention_layers = ["global", "local", "global", "local", "global", "local"]
        gptneo.config.vocab_size = 18

        mamba.config.n_layer = 12
        mamba.config.d_model = 512
        mamba.config.vocab_size = 18

        pad_token_id = 0
    elif dataset_name in ["pathfinder32", "pathfinder64", "pathfinder128", "pathfinder256"]:
        train_dataset, test_dataset, num_class, sequence_length = prepare_pathfinder_data(
            f"long_range_arena/datasets/{dataset_name}")
        hps = {
            "num_train_epochs": 10,
            # "learning_rate": 0.01,
            # "lr_scheduler_type": "constant",
        }
        gptneo.config.num_heads = 8
        gptneo.config.num_layers = 4
        gptneo.config.hidden_size = 128
        gptneo.config.intermediate_size = 128
        gptneo.config.attention_types = [[['global', 'local'], 2]]
        gptneo.config.attention_layers = ["global", "local", "global", "local"]
        gptneo.config.vocab_size = 256

        mamba.config.n_layer = 8
        mamba.config.d_model = 128
        mamba.config.vocab_size = 256
    elif dataset_name in ["imdb"]:
        train_dataset, test_dataset, num_class, sequence_length = lra_text_classification(dataset_name)
        hps = {
            # "max_steps": 80000,
            "num_train_epochs": 10,
            # "learning_rate": 0.05,
            # "weight_decay": 0.1,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            # "lr_scheduler_type": "constant",
        }

        gptneo.config.num_heads = 8
        gptneo.config.num_layers = 6
        gptneo.config.hidden_size = 512
        gptneo.config.intermediate_size = 2048
        gptneo.config.attention_types = [[['global', 'local'], 3]]
        gptneo.config.attention_layers = ["global", "local", "global", "local", "global", "local"]
        gptneo.config.vocab_size = 129

        mamba.config.n_layer = 12
        mamba.config.d_model = 512
        mamba.config.vocab_size = 129

        pad_token_id = 0
    elif dataset_name in ["mnist", "cifar10"]:
        train_dataset, test_dataset, num_class, sequence_length = lra_pixel_image_classification(dataset_name)
        hps = {
            "num_train_epochs": 10,
            # "learning_rate": 0.01,
            # "lr_scheduler_type": "constant",
        }
        gptneo.config.num_heads = 4
        gptneo.config.num_layers = 3
        gptneo.config.hidden_size = 64
        gptneo.config.intermediate_size = 128
        gptneo.config.attention_types = [[["global"], 3]]
        gptneo.config.attention_layers = ["global", "global", "global"]
        gptneo.config.vocab_size = 256

        mamba.config.n_layer = 6
        mamba.config.d_model = 64
        mamba.config.vocab_size = 256

    else:
        raise ValueError("Invalid dataset name")
    print(gptneo.config)
    print(mamba.config)
    return train_dataset, test_dataset, num_class, sequence_length, hps, gptneo.config, mamba.config, pad_token_id
