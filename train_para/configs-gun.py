DATASET_DIR = "dataset/gun/"
DATASET_FILES = [
    "2022-01-13-gun-sample-junming"
]  # , "2022-02-17-gun-sample-briony koji"]

SPLIT_RATIO = [0.8, 0.1, 0.1]

trainin_args = {
    "sequence-7b-regression": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 16,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-13b-regression": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
}
