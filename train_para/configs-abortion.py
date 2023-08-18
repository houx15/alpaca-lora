DATASET_DIR = "dataset/abortion/"
DATASET_FILES = ["2022-10-15-abortion-aggregated"]

SPLIT_RATIO = [0.8, 0.1, 0.1]


trainin_args = {
    "sequence-7b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 14,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
}
