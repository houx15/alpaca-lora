DATASET_DIR = "dataset/china/favorability/"
DATASET_FILES = [
    "2020-08-11-china-sample-favorability-en-ground-truth",
    "2020-08-11-china-sample-favorability-en-karla",
]

SPLIT_RATIO = [0.8, 0.1, 0.1]

trainin_args = {
    "sequence-7b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 4,
        "learning_rate": 1e-4,
        "cutoff_len": 256,
    },
    "sequence-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 6,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
}