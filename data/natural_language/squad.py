from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPTNeoForCausalLM


#['xquad.ar', 'xquad.de', 'xquad.el', 'xquad.en', 'xquad.es', 'xquad.hi', 'xquad.ro', 'xquad.ru', 'xquad.th', 'xquad.tr', 'xquad.vi', 'xquad.zh']
def get_squad_dataset(tokenizer, task_name):
    if task_name == "1":
        squad = load_dataset("google/xquad", 'xquad.ar')["validation"]
    elif task_name == "2":
        squad = load_dataset("google/xquad", 'xquad.de')["validation"]
    elif task_name == "3":
        squad = load_dataset("google/xquad", 'xquad.el')["validation"]
    elif task_name == "4":
        squad = load_dataset("google/xquad", 'xquad.en')["validation"]
    elif task_name == "5":
        squad = load_dataset("google/xquad", 'xquad.es')["validation"]
    elif task_name == "6":
        squad = load_dataset("google/xquad", 'xquad.hi')["validation"]
    elif task_name == "7":
        squad = load_dataset("google/xquad", 'xquad.ro')["validation"]
    elif task_name == "8":
        squad = load_dataset("google/xquad", 'xquad.ru')["validation"]
    elif task_name == "9":
        squad = load_dataset("google/xquad", 'xquad.th')["validation"]
    elif task_name == "10":
        squad = load_dataset("google/xquad", 'xquad.tr')["validation"]
    elif task_name == "11":
        squad = load_dataset("google/xquad", 'xquad.vi')["validation"]
    elif task_name == "12":
        squad = load_dataset("google/xquad", 'xquad.zh')["validation"]

    def reformat_dataset(dataset_):
        dataset_ = dataset_.map(lambda entry: {
            'question': f"Based on given context \"{entry['context']}\" Answer the following question: \"{entry['question']}\"",
            'answer': entry['answers']['text'][0]
        })
        dataset_ = dataset_.remove_columns(['context', 'answers', 'id'])
        return dataset_

    squad = reformat_dataset(squad)

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

    squad = squad.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    squad = squad.train_test_split(test_size=0.2, seed=42)
    train_dataset = squad["train"]
    eval_dataset = squad["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return train_dataset, eval_dataset, data_collator

# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/gpt-neo-1.3B"
# )  # GPT-Neo tokenizer
#
# dataset, eval_dataset, data_collator = get_xquad_dataset(tokenizer, "1")
# print(dataset)
# print(eval_dataset)
