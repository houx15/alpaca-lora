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
        "num_epochs": 16,
        "learning_rate": 1e-4,
        "cutoff_len": 256,
    },
    "sequence-13b-binary": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 6,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-binary": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 4,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        "logging_steps": 40,
        "eval_steps": 40,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-binary": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
}
