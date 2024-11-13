from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def get_alpaca_dataset(tokenizer):
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = alpaca.train_test_split(test_size=0.1, seed=1111)
    alpaca = alpaca.flatten()

    block_size = 512
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized_alpaca = alpaca.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        # remove_columns=alpaca["train"].column_names,
    )

    # TODO eval using bleu score on the outputs when prompted with the instruction

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = tokenized_alpaca["train"]
    eval_dataset = tokenized_alpaca["test"]

    return dataset, eval_dataset, data_collator
