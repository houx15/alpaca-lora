import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
)
import numpy as np
import pandas as pd

import json

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    TaskType,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    LlamaForSequenceClassification,
)

from utils.prompter import Prompter
from utils.utils import result_translator, sentence_cleaner

from tqdm import tqdm
import time


def train(
    # model/data params
    topic: str,
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(input, label, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            input,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = float(label)

        return result

    def generate_and_tokenize_prompt(data_point):
        data_point["text"] = sentence_cleaner(data_point["text"])
        tokenized_data = tokenize(data_point["text"], data_point["label"])
        return tokenized_data

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    print(config)
    model = get_peft_model(model, config)

    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    # train_pd = pd.read_csv(f"{data_path}/train-regression.csv")
    # train_data = load_dataset("json", data_files=f"{data_path}/train.json")
    # val_data = load_dataset("json", data_files=f"{data_path}/val.json")
    # test_data = load_dataset("json", data_files=f"{data_path}/test.json")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"{data_path}/train-regression.csv",
            "validate": f"{data_path}/validate-regression.csv",
            "test": f"{data_path}/test-regression.csv",
        },
    )
    # else:
    #     data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None

    # dataset = dataset.rename_column("label", "labels")
    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = dataset["validate"].shuffle().map(generate_and_tokenize_prompt)
    test_data = dataset["test"].shuffle().map(generate_and_tokenize_prompt)

    train_data = train_data.remove_columns(["label"])
    val_data = val_data.remove_columns(["label"])
    test_data = test_data.remove_columns(["label"])

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=10,
            save_steps=50,
            output_dir=output_dir,
            save_total_limit=3,
            label_names=["labels"],
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer, return_tensors="pt"
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    """
    evaluate
    """
    model.eval()
    error_analysis = {}

    def regression_metrics_compute(pred):
        # error_analysis = {}
        labels = pred.label_ids
        preds = (
            pred.predictions[0]
            if isinstance(pred.predictions, tuple)
            else pred.predictions
        )
        preds = np.squeeze(preds)

        rmse = mean_squared_error(labels, preds, squared=False)

        integerized_preds = np.around(preds)
        integerized_rmse = mean_squared_error(
            labels, integerized_preds, squared=False
        )

        diff = np.subtract(integerized_preds, labels)
        # when integerized_pred equals to label, assume there diff is 0
        label_precision_preds = np.where(diff, preds, labels)
        label_precision_rmse = mean_squared_error(
            labels, label_precision_preds, squared=False
        )
        for idx, x in np.ndenumerate(labels):
            preds_set = error_analysis.get(x, np.array(0))
            preds_set = np.append(preds_set, preds[idx])
            error_analysis[x] = preds_set

        return {
            "rmse": rmse,
            "integerized_rmse": integerized_rmse,
            "label_precision_rmse": label_precision_rmse,
        }

    predictor = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            per_device_eval_batch_size=micro_batch_size,
            do_train=False,
            logging_steps=10,
            output_dir=output_dir,
            label_names=["labels"],
        ),
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer, return_tensors="pt"
        ),
        compute_metrics=regression_metrics_compute,
    )

    pred = trainer.predict(test_dataset=test_data)
    print(pred.metrics)

    prediction = pred.predictions[0].flatten()
    prediction = np.clip(prediction, -2, 2)

    rmse_dict = {}

    # creterion = SmoothL1Loss()

    for k, v in pred.metrics.items():
        print(f"{k}:    {v}")
    for k, s in error_analysis.items():
        true = np.ones(s.shape) * k
        rmse = mean_squared_error(true, s, squared=False)
        rmse_dict[k] = rmse
        # true = torch.from_numpy(true)
        # s = torch.from_numpy(s)
        # huber = creterion(s, true)
        print(f"{k}:    rmse-{rmse}")
        # log(log_file, f'{str(k)}:    rmse-{str(rmse)}; huber-{huber}')

    torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(train)
