from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling


import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import numpy as np
from transformers import pipeline, AutoTokenizer
from transformers import GPTNeoForCausalLM
from mamba_ssm import MambaLMHeadModel
from tqdm import tqdm

import pandas as pd
from datasets import Dataset


def get_model_and_tokenizer(model_name):
    device = "cuda"
    dtype = torch.float16

    print(f"Loading model {model_name}")
    is_mamba = model_name.startswith("state-spaces/mamba-")
    if is_mamba:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": device}, torch_dtype=dtype
        )
    model.eval()
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    model_name,
    prompt=None,
    genlen=100,
    temperature=1.0,
    topk=1,
    topp=1.0,
    minp=0.0,
    repetition_penalty=1.0,
    batch=1,
):
    is_mamba = model_name.startswith("state-spaces/mamba-")

    repeats = 3
    device = "cuda"
    dtype = torch.float16

    torch.random.manual_seed(0)
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + genlen

    if is_mamba:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            min_p=minp,
            repetition_penalty=repetition_penalty,
        )
    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            repetition_penalty=repetition_penalty,
        )
    out = fn()
    if prompt is not None:
        res = tokenizer.batch_decode(out.sequences.tolist())

        return res


def get_mix_dataset(tokenizer, slice=None):
    ptb = load_dataset("ptb_text_only")
    # ptb = ptb.train_test_split(test_size=0.1, seed=1111)
    ptb = ptb.flatten()

    try:
        mix_dataset = load_from_disk("data/mamba_gptneo_mix_ptb/")
        print("LOADED")
    except:
        gptneo_model, gpneo_tokenizer = get_model_and_tokenizer(
            "EleutherAI/gpt-neo-125m"
        )
        mamba_model, mamba_tokenizer = get_model_and_tokenizer(
            "state-spaces/mamba-130m"
        )
        generator_gptneo = pipeline(
            "text-generation", model=gptneo_model, tokenizer=gpneo_tokenizer
        )

        res_length = 30

        def generate_mamba(prompt):
            text = generate_text(
                model=mamba_model,
                tokenizer=mamba_tokenizer,
                model_name="state-spaces/mamba-130m",
                prompt=prompt,
                batch=1,  # TODO ?
                genlen=res_length,
            )
            return text[0]

        def generate_gptneo(prompt):
            text = generator_gptneo(
                prompt,
                do_sample=True,
                min_new_tokens=res_length,
                max_new_tokens=res_length,
                pad_token_id=generator_gptneo.tokenizer.eos_token_id,
            )  # TODO ?
            return text[0]["generated_text"]

        k = 4  # Number of words to prompt with

        completions = []
        slice_meta = []
        for i, s in enumerate(tqdm(ptb["validation"])):
            s_trunc = " ".join(s["sentence"].split()[:k])
            res_mamba = generate_mamba(s_trunc)
            completions.append(res_mamba)
            slice_meta.append("mamba")

            res_gptneo = generate_gptneo(s_trunc)
            completions.append(res_gptneo)
            slice_meta.append("gptneo")

        df = pd.DataFrame({"sentence": completions, "slice": slice_meta})
        mix_dataset = Dataset.from_pandas(
            df.rename(columns={0: "sentence"}), split="train"
        )
        mix_dataset = mix_dataset.train_test_split(test_size=0.3, seed=1111)
        mix_dataset.save_to_disk("data/mamba_gptneo_mix_ptb/")

    # NOTE filter by slice
    if slice is None:
        slices = ["mamba", "gptneo"]
    else:
        slices = [slice]

    mix_dataset = mix_dataset.filter(lambda x: x["slice"] in slices)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["sentence"]])

    tokenized_mix = mix_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=mix_dataset["train"].column_names,
    )
    block_size = 512

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

    lm_dataset = tokenized_mix.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = lm_dataset["train"]
    eval_dataset = lm_dataset["test"]

    return dataset, eval_dataset, data_collator
