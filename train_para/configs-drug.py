DATASET_DIR = "dataset/drug/"
DATASET_FILES = ["2023-02-13-drug-legalization-aggregated"]

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
    "prompt-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 8,
        "num_epochs": 15,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 20,
        "eval_steps": 20,
        "save_steps": 40,
    },
}
