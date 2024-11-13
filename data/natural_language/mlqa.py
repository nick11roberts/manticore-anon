from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM


def get_mlqa_dataset(tokenizer, task_name):
    if task_name == "1":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.es.es")
    elif task_name == "2":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.ar.ar")
    elif task_name == "3":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.vi.vi")
    elif task_name == "4":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.zh.zh")
    elif task_name == "5":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.hi.hi")
    elif task_name == "6":
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.de.de")
    else:
        ds_mlqa = load_dataset("facebook/mlqa", "mlqa.en.en")

    def reformat_dataset(dataset_):
        dataset_ = dataset_.map(lambda entry: {
            'question': f"Based on given context \"{entry['context']}\" Answer the following question: \"{entry['question']}\"",
            'answer': entry['answers']['text'][0]
        })
        # only keep this 2 entry
        dataset_ = dataset_.remove_columns(['context', 'answers', 'id'])
        return dataset_

    mlqa_train, mlqa_eval = map(reformat_dataset, [ds_mlqa['test'], ds_mlqa['validation']])

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
    
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = alpaca.train_test_split(test_size=0.2, seed=42)
    alpaca = alpaca.flatten()

    def tokenize_function_a(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized_alpaca = alpaca.map(
        tokenize_function_a,
        batched=True,
        num_proc=4,
        # remove_columns=alpaca["train"].column_names,
    )

    alpaca_train = tokenized_alpaca["train"]
    alpaca_eval = tokenized_alpaca["test"]

    mlqa_train_tokenized = mlqa_train.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )
    mlqa_eval_tokenized = mlqa_eval.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    train_length = min(len(mlqa_train_tokenized), len(alpaca_train))
    eval_length = min(len(mlqa_eval_tokenized), len(alpaca_eval))

    alpaca_train = alpaca_train.shuffle(seed=42).select(range(train_length))
    alpaca_eval = alpaca_eval.shuffle(seed=42).select(range(eval_length))
    mlqa_train_tokenized = mlqa_train_tokenized.shuffle(seed=42).select(range(train_length))
    mlqa_eval_tokenized = mlqa_eval_tokenized.shuffle(seed=42).select(range(eval_length))

    dataset = concatenate_datasets([alpaca_train, mlqa_train_tokenized]).shuffle(seed=42)
    eval_dataset = concatenate_datasets([alpaca_eval, mlqa_eval_tokenized]).shuffle(seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return dataset, eval_dataset, data_collator

#
# gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/gpt-neo-1.3B"
# )  # GPT-Neo tokenizer
#
# dataset, eval_dataset, data_collator = get_mlqa_dataset(tokenizer)
# print(dataset)
# print(eval_dataset)
# print(data_collator)
