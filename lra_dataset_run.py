import functools
import itertools
import json
import os
import time
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import long_range_arena.dataloaders.lra as lra

import sys
import fire
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, LlamaForCausalLM, GPTNeoForSequenceClassification
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import DataCollatorForLanguageModeling
from mamba_ssm import MambaLMHeadModel
from based.models.gpt import GPTLMHeadModel
from models.manticore import ManticoreConfig, ManticoreForCausalLM, ManticoreForSequenceClassification
from datasets import load_dataset, Dataset, DatasetDict
import argparse

from callbacks.wandb import WandbAlphasCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# choice: imdb, yelp_polarity, ag_news
def lra_text_classification(dataset_name: str):
    if dataset_name != "imdb":
        raise ValueError("Invalid dataset name")
    data = lra.IMDB(dataset_name, l_max=2048)
    data.setup()
    dataset, tokenizer, vocab = data.process_dataset()
    # print(dataset)
    print(len(vocab))
    print(vocab.get_itos())
    print(vocab.get_stoi())
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
    sequence_length = max(max(len(i) for i in train_dataset["input_ids"]),
                          max(len(i) for i in test_dataset["input_ids"]))
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


# To download those dataset, https://storage.googleapis.com/long-range-arena/lra_release.gz. Then extract the file
# File structure:
# long_range_arena/datasets/
# ├── listops-1000
# ├── pathfinder128
# ├── pathfinder256
# ├── pathfinder32
# └── pathfinder64
def main(
        d_model=768,
        n_mixtures=4,
        fixed_alphas=False,
):
    config = ManticoreConfig(
        d_model=d_model, n_mixtures=n_mixtures, fixed_alphas=fixed_alphas
    )

    mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-1.3B"
    )  # GPT-Neo tokenizer

    tokenizer.pad_token = tokenizer.eos_token

    ########################################### DIFFERENT DATASETS ############################################
    # train_dataset, test_dataset, num_class = lra_pixel_image_classification("mnist")
    # train_dataset, test_dataset, num_class, sequence_length = lra_pixel_image_classification("cifar10")

    train_dataset, test_dataset, num_class, sequence_length = lra_text_classification("imdb")
    # train_dataset, test_dataset, num_class, sequence_length = lra_text_classification("yelp_polarity", tokenizer)
    # train_dataset, test_dataset, num_class, sequence_length = lra_text_classification("ag_news", tokenizer)

    # train_dataset, test_dataset, num_class, sequence_length = prepare_listops_data('long_range_arena/datasets/listops-1000')

    # train_dataset, test_dataset, num_class, sequence_length = prepare_pathfinder_data("long_range_arena/datasets/pathfinder32")
    # train_dataset, test_dataset, num_class, sequence_length = prepare_pathfinder_data("long_range_arena/datasets/pathfinder64")
    # train_dataset, test_dataset, num_class, sequence_length = prepare_pathfinder_data("long_range_arena/datasets/pathfinder128")
    # train_dataset, test_dataset, num_class, sequence_length = prepare_pathfinder_data("long_range_arena/datasets/pathfinder256")
    ##########################################################################################################
    print(train_dataset[0])
    print("train_dataset: ", train_dataset, "\n", "test_dataset: ", test_dataset, "\n",
          "num_class: ", num_class, "sequence_length: ", sequence_length)

    model = ManticoreForSequenceClassification(config=config, num_classes=num_class, mamba=mamba, gptneo=gptneo).to(
        device=device)
    model.config.pad_token_id = 50256
    print(model.config.pad_token_id)
    training_args = TrainingArguments(
        output_dir="trash",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=10,  # TODO
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # callbacks=[WandbAlphasCallback(freq=1)],
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
