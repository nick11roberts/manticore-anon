from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def get_ptb_dataset(tokenizer):
    ptb = load_dataset("ptb_text_only")
    # ptb = ptb.train_test_split(test_size=0.1, seed=1111)
    ptb = ptb.flatten()

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["sentence"]])

    tokenized_ptb = ptb.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=ptb["train"].column_names,
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

    lm_dataset = tokenized_ptb.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = lm_dataset["train"]
    eval_dataset = lm_dataset["test"]

    return dataset, eval_dataset, data_collator
