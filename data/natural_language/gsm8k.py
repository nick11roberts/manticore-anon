from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM


def get_gsm8k_dataset(tokenizer):
    # There are 2 files - 'main' and 'socratic'
    gsm8k = load_dataset("gsm8k", "main", split="train")
    gsm8k = gsm8k.train_test_split(test_size=0.1, seed=1111)
    gsm8k = gsm8k.flatten()

    block_size = 512
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(
            example["question"],
            example["answer"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized_gsm8k = gsm8k.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    # TODO eval using bleu score on the outputs when prompted with the instruction

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = tokenized_gsm8k["train"]
    eval_dataset = tokenized_gsm8k["test"]

    return dataset, eval_dataset, data_collator


# gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/gpt-neo-1.3B"
# )  # GPT-Neo tokenizer

# dataset, eval_dataset, data_collator = get_gsm8k_dataset(tokenizer)
# print(dataset)
# print(eval_dataset)
# print(data_collator)
