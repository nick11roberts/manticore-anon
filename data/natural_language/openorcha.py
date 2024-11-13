from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def get_openorcha_dataset(tokenizer):
    orcha_set = load_dataset("atsushi3110/cross-lingual-openorcha-830k-en-ja")
    selected_set = orcha_set['train'].filter(lambda example: example['system_prompt/en'] == "")
    random_10000 = selected_set.shuffle(seed=42).select(range(10000))
    random_10000 = random_10000.train_test_split(test_size=0.2, seed=42)

    def reformat_dataset(dataset_):
        dataset_ = dataset_.map(lambda entry: {
            'question': entry["question/ja"],
            'answer': entry['response/ja']
        })
        dataset_ = dataset_.remove_columns(['question/en', 'response/en', 'question/ja', 'response/ja', 'system_prompt/en', 'id/en'])
        return dataset_

    train_set = reformat_dataset(random_10000['train'])
    # print(train_set[0])
    eval_set = reformat_dataset(random_10000['test'])

    block_size = 512

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(
            example["question"],
            example["answer"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    trained_tokenized = train_set.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    eval_tokenized = eval_set.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return trained_tokenized, eval_tokenized, data_collator
#
# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/gpt-neo-1.3B"
# )  # GPT-Neo tokenizer
#
# dataset, eval_dataset, data_collator = get_openorcha_dataset(tokenizer)
# print(dataset)
# print(eval_dataset)