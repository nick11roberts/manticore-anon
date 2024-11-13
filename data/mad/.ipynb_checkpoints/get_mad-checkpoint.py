import yaml

from mad.configs import MADConfig, MADModelConfig
from mad.data import generate_data
from datasets import load_dataset, Dataset, DatasetDict


def get_mad(task_name):

    # with open(f"mad/mad_default_config/{args.get('task')}.yml") as f:
    #     default_config = yaml.safe_load(f)
    # print(default_config['baseline'])
    # # update args with default config
    # for k, v in default_config['baseline'].items():
    #     args[k] = v

    # mad_config = MADConfig()
    # mad_config.update_from_kwargs(args)

    with open(f"mad/mad_default_config/{task_name}.yml") as f:
        default_config = yaml.safe_load(f)

    mad_config = MADConfig(task=task_name)
    mad_config.update_from_kwargs(default_config["baseline"])
    print(mad_config)
    # quit()

    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path,
        test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples,
        num_workers=mad_config.num_data_workers,
    )

    ###########
    input_data = [item[0] for item in data["train"]]
    label = [item[1] for item in data["train"]]
    dataset = Dataset.from_dict({"input_ids": input_data, "labels": label})
    # print(dataset)
    # print(dataset[0])
    # quit()

    input_data_test = [item[0] for item in data["test"]]
    label_test = [item[1] for item in data["test"]]
    eval_dataset = Dataset.from_dict(
        {"input_ids": input_data_test, "labels": label_test}
    )
    print(eval_dataset)

    return dataset, eval_dataset, mad_config
