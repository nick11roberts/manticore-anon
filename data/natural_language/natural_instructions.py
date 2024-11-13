from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
import os
import json


def check_NI():
    path = "../../datasets/natural-instructions-2.8/tasks"
    # list all json files in the path
    files = os.listdir(path)
    task_type = {}
    count_non_english_qa = 0
    non_english_qa_files = []

    count_partially_non_english_qa = 0
    partially_non_english_qa_files = []

    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(path, file)) as f:
                data = json.load(f)
                curr_type = data["Categories"][0]
                task_type[curr_type] = task_type.get(curr_type, 0) + 1
                if curr_type == "Question Answering":
                    if data["Input_language"][0] != "English" and data["Output_language"][0] != "English":
                        count_non_english_qa += 1
                        non_english_qa_files.append(os.path.join(path, file))
                    elif data["Input_language"][0] != "English" or data["Output_language"][0] != "English":
                        count_partially_non_english_qa += 1
                        partially_non_english_qa_files.append(os.path.join(path, file))

    # Write the non-English QA file paths to a new JSON file
    with open("../../datasets/natural-instructions-2.8/non_english_qa_files.json", "w") as outfile:
        json.dump(non_english_qa_files, outfile, indent=4)
    with open("../../datasets/natural-instructions-2.8/partially_non_english_qa_files.json", "w") as outfile:
        json.dump(partially_non_english_qa_files, outfile, indent=4)

    # print(task_type)
    # print(count_non_english_qa)


def get_NI_dataset(tokenizer, task_name):
    block_size = 512
    tokenizer.pad_token = tokenizer.eos_token
    # split task_name by character
    list_task_name = list(task_name)
    if len(list_task_name) != 1:
        file_path = []
        with open("../../datasets/natural-instructions-2.8/non_english_qa_files.json") as f:
            non_english_qa_files = json.load(f)
            for task in list_task_name:
                file_path.append(non_english_qa_files[int(task) - 1])
        dataset_dict_ = {
            'question': [],
            'answer': []
        }
        for file in file_path:
            with open(file) as f:
                data = json.load(f)
                all_instances = data["Instances"]
                for instance in all_instances:
                    dataset_dict_['question'].append(instance["input"])
                    dataset_dict_['answer'].append(instance["output"][0])
        temp_dataset = Dataset.from_dict(dataset_dict_).shuffle(seed=42)
        split_dataset = temp_dataset.train_test_split(test_size=0.2, seed=42)
        dataset_ = split_dataset['train']
        eval_dataset = split_dataset['test']

        def tokenize_function(example):
            return tokenizer(
                example["question"],
                example["answer"],
                truncation=True,
                padding="max_length",
                max_length=block_size,
            )

        dataset_ = dataset_.map(
            tokenize_function,
            batched=True,
            num_proc=4,
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        return dataset_, eval_dataset, data_collator

    else:
        with open("../../datasets/natural-instructions-2.8/non_english_qa_files.json") as f:
            non_english_qa_files = json.load(f)
            file_path = non_english_qa_files[int(task_name) - 1]
            print(file_path)
        dataset_dict_ = {
            'question': [],
            'answer': []
        }
        with open(file_path) as f:
            data = json.load(f)
            all_instances = data["Instances"]
            for instance in all_instances:
                dataset_dict_['question'].append(instance["input"])
                dataset_dict_['answer'].append(instance["output"][0])
        temp_dataset = Dataset.from_dict(dataset_dict_)
        split_dataset = temp_dataset.train_test_split(test_size=0.2, seed=42)
        dataset_ = split_dataset['train']
        eval_dataset = split_dataset['test']

        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca = alpaca.train_test_split(test_size=0.2, seed=42)
        alpaca = alpaca.flatten()

        # print(dataset_[0]['question'])

        def tokenize_function(example):
            return tokenizer(
                example["question"],
                example["answer"],
                truncation=True,
                padding="max_length",
                max_length=block_size,
            )

        dataset_ = dataset_.map(
            tokenize_function,
            batched=True,
            num_proc=4,
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
        )

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

        min_length = min(len(dataset_), len(alpaca_train))
        dataset_ = dataset_.shuffle(seed=42).select(range(min_length))
        alpaca_train = alpaca_train.shuffle(seed=42).select(range(min_length))
        combined_train = concatenate_datasets([dataset_, alpaca_train]).shuffle(seed=42)

        min_length_eval = min(len(eval_dataset), len(alpaca_eval))
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(min_length_eval))
        alpaca_eval = alpaca_eval.shuffle(seed=42).select(range(min_length_eval))
        combined_eval = concatenate_datasets([eval_dataset, alpaca_eval]).shuffle(seed=42)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return combined_train, combined_eval, data_collator



# check_NI()

# gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/gpt-neo-1.3B"
# )  # GPT-Neo tokenizer
#
# dataset, eval_dataset, data_collator = get_NI_dataset(tokenizer, 1)
# print(dataset)
# print(eval_dataset)
# print(data_collator)
