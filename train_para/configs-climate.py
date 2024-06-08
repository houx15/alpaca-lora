DATASET_DIR = "dataset/climate/"
# DATASET_FILES = ["2022-11-02-climate-change-sample-molardata", "2022-11-14-climate-change-sample-molardata"]
DATASET_FILES = [
    "2023-09-20-climate-change-aggregated",
    # "2023-09-20-climate-change-aggregated-oversample",
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
    "sequence-13b-binary": {
        "batch_size": 16,
        "micro_batch_size": 16,
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
    "prompt-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 16,
        "num_epochs": 30,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 20,
        "eval_steps": 20,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-binary": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
}


def hp_space_optuna(trial):
    return {
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32]
        ),
    }
