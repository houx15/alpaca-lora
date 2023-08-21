import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import importlib

import fire

from data_process import DataProcess
from model import LlamaModel

model_dict = {
    "7b": "chainyo/alpaca-lora-7b",
    "13b": "yahma/llama-13b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf"
}

peft_dict = {
    "7b": "PEFT/alpaca-lora-7b", 
    "13b": "PEFT/alpaca-13b-lora"
}


def run(
    topic: str,
    dimension: str = None,
    do_train: bool = False,
    do_eval: bool = False,
    dataset_update: bool = False,
    augmentation: bool = False,
    task_type: str = "regression",  # options: regression or binary
    strategy: str = "sequence",  # options: sequence or generation
    model_type: str = "7b",  # options: 7b or 13b
    labeller: str = None,
    peft: bool = True,
    use_pretrained_peft_weights: bool = False,
    eval_model_path: str = None,
    parameter_search: bool = False,
    output_dir: str = None,
    log_dir: str = None,
    data_path: str = None,
):
    """
    Entry of the project.
    config should contain the following params:
    - topic: str, the topic for online-opinion. Topic including gun, china and us is now built-in
    - task_type: str, train task type, regression or binary
    - config: str, the path for config file. Use configs-{topic}.py  (or 'configs-{topic}-{dimension}' when dimension is specified) in the 'configs' directory by default
    """
    config_module = None
    config_module_name = f"train_para.configs-{topic}"
    if dimension is not None:
        config_module_name += f"-{dimension}"
    config_module = importlib.import_module(config_module_name)

    if config_module is None:
        raise ValueError(
            "You need to specify config file or topic info to run this project"
        )

    if augmentation and not dataset_update:
        print(
            "Warning: you need to set dataset_update True to perform data augmentation"
        )

    if output_dir is None:
        output_dir = f"output/{strategy}/{model_type}/{topic}/{task_type}"
    if log_dir is None:
        log_dir = f"logs/{strategy}/{model_type}/{topic}/{task_type}"

    augment_args = None
    if hasattr(config_module, "augment_args"):
        augment_args = config_module.augment_args

    undersampling_strategy = None
    if hasattr(config_module, "undersampling_strategy"):
        undersampling_strategy = config_module.undersampling_strategy

    data_process = DataProcess(
        config_module.DATASET_DIR,
        dataset_files=config_module.DATASET_FILES,
        task_type=task_type,
        force_update=dataset_update,
        augmentation=augmentation,
        split_ratio=config_module.SPLIT_RATIO,
        augment_args=augment_args,
        labeller=labeller,
        undersampling_strategy=undersampling_strategy,
    )

    train_df, validate_df, test_df, total_df = data_process.get_dataset()

    peft_weights = peft_dict[model_type]
    if do_eval:
        if eval_model_path is None:
            eval_model_path = output_dir
        use_pretrained_peft_weights = True
        peft_weights = eval_model_path

    model = LlamaModel(
        topic=topic,
        strategy=strategy,
        task_type=task_type,
        base_model=model_dict[model_type],
        model_type=model_type,
        param_dict=config_module.trainin_args.get(
            f"{strategy}-{model_type}-{task_type}", {}
        ),
        peft=peft,
        peft_weights=peft_weights if use_pretrained_peft_weights else None,
        output_dir=output_dir,
        log_dir=log_dir,
    )

    if do_train:
        model.train(parameter_search=parameter_search)
        model.eval()
    elif do_eval:
        model.eval()


if __name__ == "__main__":
    fire.Fire(run)
