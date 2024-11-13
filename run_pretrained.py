import sys
import torch
import fire
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
# from mamba_ssm import MambaLMHeadModel
from based.models.gpt import GPTLMHeadModel
from models.manticore import ManticoreConfig, ManticoreForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from mad.configs import MADConfig, MADModelConfig
from mad.paths import make_log_path
from mad.data import generate_data
from mad.model import PLModelWrap
from mad.registry import task_registry, layer_registry
import argparse
from transformers import MambaConfig, MambaForCausalLM

from callbacks.wandb import WandbAlphasCallback

device = "cuda"

# TODO (Srinath) - Cleanup args!
def get_args():
    parser = argparse.ArgumentParser()

    # arch settings
    parser.add_argument(
        "--arch",
        type=str,
        default="supernet",
        help="model to train",
    )

    # task settings:
    parser.add_argument(
        "--task",
        type=str,
        default="in-context-recall",
        choices=list(task_registry.keys()),
        help="task to train model on",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=16, help="size of token vocabulary"
    )
    parser.add_argument(
        "--seq_len", type=int, default=128, help="length of input sequences"
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=12_800,
        help="number of training examples",
    )
    parser.add_argument(
        "--num_test_examples", type=int, default=1_280, help="number of test examples"
    )
    parser.add_argument(
        "--frac_noise",
        type=float,
        default=0.0,
        help="fraction of input sequence that is noise",
    )
    parser.add_argument(
        "--noise_vocab_size", type=int, default=0, help="size of noise token vocabulary"
    )
    parser.add_argument(
        "--num_tokens_to_copy",
        type=int,
        default=0,
        help="number of tokens to copy in selective-copying",
    )
    parser.add_argument(
        "--k_motif_size",
        type=int,
        default=1,
        help="number of adjacent tokens that together form a key in fuzzy in-context recall",
    )
    parser.add_argument(
        "--v_motif_size",
        type=int,
        default=1,
        help="number of adjacent tokens that together form a value in fuzzy in-context recall",
    )
    parser.add_argument(
        "--multi_query",
        action="store_true",
        default=True,
        help="if True, multi-query variant of in-context recall tasks is used",
    )

    args = vars(parser.parse_args())

    # make sure we select the correct model backbone!
    if args["task"] in {"compression"} and args["backbone"] != "autoencoder":
        print(
            f'Setting model backbone to "autoencoder", which is required for the compression task!'
        )
        args["backbone"] = "autoencoder"

    return args

def main(
    d_model=768,
    n_mixtures=4,
    fixed_alphas=False,
    model_ckpt='EleutherAI/gpt-neo-125m'
):
    config = ManticoreConfig(
        d_model=d_model, n_mixtures=n_mixtures, fixed_alphas=fixed_alphas
    )

    if model_ckpt == 'state-spaces/mamba-130m-hf':
        # original Mamba source
        # mamba = MambaLMHeadModel.from_pretrained(model_ckpt)
        # model = mamba.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(
        # "EleutherAI/gptx-20b" # it's a diff tokenizer
        # )  

        # hf version
        mamba = MambaForCausalLM.from_pretrained(model_ckpt)
        model = mamba.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt
        )  
    
    if model_ckpt == 'EleutherAI/gpt-neo-125m':
        gptneo = GPTNeoForCausalLM.from_pretrained(model_ckpt)
        model = gptneo.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt
        )  

    print(model)
    print(config)

    ########## MAD Dataset ########
    args = get_args()

    ########## load default config from yml files ########
    with open(f"mad/mad_default_config/{args.get('task')}.yml") as f:
        default_config = yaml.safe_load(f)
    print(default_config['baseline'])
    # update args with default config
    for k, v in default_config['baseline'].items():
        args[k] = v

    mad_config = MADConfig()
    mad_config.update_from_kwargs(args)
    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path,
        test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples,
        num_workers=mad_config.num_data_workers,
    )

    print(mad_config)
    # change the tuple to dataset, input is data["train"][i][0], label is data["train"][i][1]
    input_data = [item[0] for item in data["train"]]
    label = [item[1] for item in data["train"]]
    train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
    print(train_dataset)
    input_data_test = [item[0] for item in data["test"]]
    label_test = [item[1] for item in data["test"]]
    test_dataset = Dataset.from_dict(
        {"input_ids": input_data_test, "labels": label_test}
    )

    from transformers import DataCollatorForLanguageModeling
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="trash",
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=1,
        learning_rate=1e-40,
        weight_decay=0.01,
        num_train_epochs=1,  # TODO
        save_strategy="epoch",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[WandbAlphasCallback(freq=1)],
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main(model_ckpt="state-spaces/mamba-130m-hf"))
    fire.Fire(main(model_ckpt="EleutherAI/gpt-neo-125m"))
