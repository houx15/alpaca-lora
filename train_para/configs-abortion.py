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
    "sequence-13b-binary": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-chat-regression": {
        "batch_size": 64,
        "micro_batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 40,
        "save_steps": 40,
    },
    "sequence-llama-2-7b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 25,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 30,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "sequence-llama-2-13b-binary": {
        "batch_size": 16,
        "micro_batch_size": 16,
        "num_epochs": 30,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
    },
    "prompt-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 8,
        "num_epochs": 15,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 40,
        "save_steps": 40,
    },
    "prompt-llama-2-13b-regression": {
        "batch_size": 32,
        "micro_batch_size": 16,
        "num_epochs": 15,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 40,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-regression": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 20,
        "eval_steps": 20,
        "save_steps": 40,
    },
    "generation-llama-2-13b-chat-binary": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 15,
        "learning_rate": 1e-3,
        "cutoff_len": 256,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
}


def hp_space_optuna(trial):
    return {
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.01, 0.03, 0.05, 0.1, 0.2]
        ),
        "warmup_steps": trial.suggest_categorical(
            "warmup_steps", [20, 40, 50, 60, 100, 200]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32]
        ),
    }
