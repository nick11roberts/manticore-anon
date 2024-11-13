from datasets import load_dataset, ReadInstruction
from transformers import DataCollatorForLanguageModeling


def get_fineweb_dataset(tokenizer):
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        streaming=False,
        split="train[:1%]",
    )
    fineweb = fineweb.flatten()

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    tokenized_fineweb = fineweb.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=fineweb.column_names,
    )
    block_size = 1024

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

    lm_dataset = tokenized_fineweb.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return lm_dataset, None, data_collator
