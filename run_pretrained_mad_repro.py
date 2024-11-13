import sys
import torch
import fire
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, GPTNeoConfig, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
# from mamba_ssm import MambaLMHeadModel
#from based.models.gpt import GPTLMHeadModel
from models.manticore import ManticoreConfig, ManticoreForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from mad.configs import MADConfig, MADModelConfig
from mad.paths import make_log_path
from mad.data import generate_data
from mad.model import PLModelWrap
from mad.registry import task_registry, layer_registry
import argparse
#from transformers import MambaConfig, MambaForCausalLM

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

    if model_ckpt == 'EleutherAI/gpt-neo-125m':
        #gptneo = GPTNeoForCausalLM.from_pretrained(model_ckpt)

        from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

        config = GPTNeoXConfig(
            hidden_size=512, num_hidden_layers=2, num_heads=16, classifier_dropout=0, vocab_size=16, intermediate_size=512)

        gptneo = GPTNeoXForCausalLM(config)
        print(gptneo)
        #quit()
        
        #quit()

        model = gptneo.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt
        )  

        num_params = sum(p.numel() for p in model.parameters())
        print(num_params)
        #quit()

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
    print(mad_config)

    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path,
        test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples,
        num_workers=mad_config.num_data_workers,
    )

    # NOTE preprocess data for HF causal LM
    import numpy as np

    print(mad_config)
    # change the tuple to dataset, input is data["train"][i][0], label is data["train"][i][1]
    input_data = [np.append(item[0], item[1][-1]) for item in data["train"]]
    label = input_data
    train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": input_data})
    print(train_dataset)

    # print(input_data[0])
    # print(label[0])
    # quit()

    input_data_test = [np.append(item[0], item[1][-1]) for item in data["test"]]
    label_test = input_data_test
    test_dataset = Dataset.from_dict(
        {"input_ids": input_data_test, "labels": input_data_test}
    )
    # NOTE preprocess data for HF causal LM

    # print(mad_config)
    # # change the tuple to dataset, input is data["train"][i][0], label is data["train"][i][1]
    # input_data = [item[0] for item in data["train"]]
    # label = [item[1] for item in data["train"]]
    # train_dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
    # print(train_dataset)
    # input_data_test = [item[0] for item in data["test"]]
    # label_test = [item[1] for item in data["test"]]
    # test_dataset = Dataset.from_dict(
    #     {"input_ids": input_data_test, "labels": label_test}
    # )

    from transformers import DataCollatorForLanguageModeling
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    training_args = TrainingArguments(
        output_dir="trash",
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        weight_decay=0,
        num_train_epochs=10,  # TODO
        save_strategy="epoch",
        report_to="wandb",
        lr_scheduler_type="constant",
        per_device_train_batch_size=128,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        #tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #data_collator=data_collator,
        callbacks=[WandbAlphasCallback(freq=1)],
    )
    #trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    #fire.Fire(main(model_ckpt="state-spaces/mamba-130m-hf"))
    fire.Fire(main)
