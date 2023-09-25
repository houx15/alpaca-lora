DATASET_DIR = "dataset/sexual/"
DATASET_FILES = ["2023-04-02-sexual-orientation-aggregated"]

SPLIT_RATIO = [0.8, 0.1, 0.1]

trainin_args = {
    "sequence-7b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 5e-4,
        "cutoff_len": 256,
    },
    "generation-llama-2-13b-chat-regression": {
        "batch_size": 64,
        "micro_batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 1e-2,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 40,
        "save_steps": 40,
    },
}
