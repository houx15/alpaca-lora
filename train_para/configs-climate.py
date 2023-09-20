DATASET_DIR = "dataset/climate/"
# DATASET_FILES = ["2022-11-02-climate-change-sample-molardata", "2022-11-14-climate-change-sample-molardata"]
DATASET_FILES = [
    "2023-09-20-climate-change-aggregated",
    "2023-09-20-climate-change-aggregated-oversample",
]
OVERSAMPLE_FILES = ["2023-09-20-climate-change-aggregated-oversample"]

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
    "sequence-llama-2-13b-binary": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 14,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
}
