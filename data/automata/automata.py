class AutomatonDataset:
    def __init__(self, name: str, ntrain: int):
        # ntrain = 600_000  # 2048  # 600_000 # 5_000_000
        length = 100

        # TODO subclass
        if name == "abab":
            input_vocab_size = 2
            output_vocab_size = 5
        elif name == "add":
            raise NotImplementedError
        elif name == "alternating":
            raise NotImplementedError
        elif name == "cyclic":
            input_vocab_size = 2
            output_vocab_size = 5
        elif name == "dihedral":
            raise NotImplementedError
        elif name == "flipflop":
            input_vocab_size = 3
            output_vocab_size = 3
        elif name == "gridworld":
            input_vocab_size = 2
            output_vocab_size = 9
        elif name == "parity":
            input_vocab_size = 2
            output_vocab_size = 2
        elif name == "quaternion":
            input_vocab_size = 4
            output_vocab_size = 8
        elif name == "symmetric":
            raise NotImplementedError
        elif name == "permutation_reset":
            raise NotImplementedError
        else:
            raise NotImplementedError

        datapath = "data/synthseq/automata/automata.py"

        config_train = {"size": ntrain, "name": name, "length": length}
        dataset_train = load_dataset(
            datapath,
            config=config_train,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # config_train.cleanup_cache_files()

        config_val = {"size": 2048, "name": name, "length": length}
        dataset_val = load_dataset(
            datapath,
            config=config_val,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # dataset_val.cleanup_cache_files()

        config_test = {"size": 2048, "name": name, "length": length}
        dataset_test = load_dataset(
            datapath,
            config=config_test,
            download_mode="force_redownload",
            ignore_verifications=True,
        )
        # dataset_test.cleanup_cache_files()

        self.length = length
        self.ntrain = ntrain
        self.name = name
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
