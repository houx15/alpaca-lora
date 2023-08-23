import os
import sys

from typing import List

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    TaskType,
    PromptTuningConfig,
    PromptTuningInit,
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    LlamaForSequenceClassification,
)

from utils.prompter import Prompter
from utils.utils import result_translator, sentence_cleaner

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
)
from scipy import special

import numpy as np
import pandas as pd
import json

from tqdm import tqdm
import tensorboardX


class TrainingPara(object):
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
    train_on_inputs: bool = False  # if False, masks out inputs in loss
    add_eos_token: bool = True
    group_by_length: bool = (
        False  # faster, but produces an odd training loss curve
    )
    warmup_steps: int = 50
    optim: str = "adamw_torch"
    logging_steps: int = 50
    eval_steps: int = 50
    save_steps: int = 50

    def __init__(self, param_dict: dict = {}):
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.gradient_accumulation_steps = (
            self.batch_size // self.micro_batch_size
        )


class LlamaModel(object):
    def __init__(
        # model/data params
        self,
        topic: str = None,
        strategy: str = "sequence",  # sequence or generation or prompt
        task_type: str = "regression",  # regression or binary, disabled for generation
        base_model: str = "",  # the only required argument
        model_type: str = "7b",  # 7b or 13b
        data_path: str = None,
        output_dir: str = None,
        log_dir: str = None,
        param_dict: dict = {},
        load_8bit: bool = True,
        world_size: int = 1,
        peft: bool = True,
        peft_weights: str = None,
        device_map="auto",
        # TODO wandb params, unable to use as the GPU cannot access to internet
        # wandb_project: str = "",
        # wandb_run_name: str = "",
        # wandb_watch: str = "all",  # options: false | gradients | all
        # wandb_log_model: str = "true",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    ) -> None:
        self.topic = topic
        self.config = TrainingPara(param_dict)
        self.strategy = strategy
        self.task_type = task_type
        self.base_model = base_model
        self.load_8bit = load_8bit
        self.peft = peft
        self.peft_weights = peft_weights
        self.device_map = device_map
        self.resume_from_checkpoint = resume_from_checkpoint
        self.error_analysis = {}

        if data_path is None:
            data_path = (
                f"json/{topic}"
                if strategy == "generation" or strategy == "prompt"
                else f"dataset/{topic}"
            )
            # TODO adapt to dimensions in the future
            if topic == "china":
                data_path += "/favorability"

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"data path {data_path} doesn't exist")

        if output_dir is None:
            output_dir = f"output/{strategy}/{model_type}/{topic}/{task_type}"
        if log_dir is None:
            log_dir = f"logs/{strategy}/{model_type}/{topic}/{task_type}"

        self.output_dir = output_dir
        self.log_dir = log_dir

        if self.strategy == "prompt":
            prompt_template_name = "prompt-tuning"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"\nTraining model with params:\n"
                f"base_model: {base_model}\n"
                f"data_path: {data_path}\n"
                f"output_dir: {output_dir}\n"
                f"batch_size: {self.config.batch_size}\n"
                f"micro_batch_size: {self.config.micro_batch_size}\n"
                f"num_epochs: {self.config.num_epochs}\n"
                f"learning_rate: {self.config.learning_rate}\n"
                f"cutoff_len: {self.config.cutoff_len}\n"
                f"train_on_inputs: {self.config.train_on_inputs}\n"
                f"add_eos_token: {self.config.add_eos_token}\n"
                f"group_by_length: {self.config.group_by_length}\n"
                # f"wandb_project: {wandb_project}\n"
                # f"wandb_run_name: {wandb_run_name}\n"
                # f"wandb_watch: {wandb_watch}\n"
                # f"wandb_log_model: {wandb_log_model}\n"
                f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                f"prompt template: {prompt_template_name}\n"
            )
            if self.strategy != "prompt":
                print(
                    f"lora_r: {self.config.lora_r}\n"
                    f"lora_alpha: {self.config.lora_alpha}\n"
                    f"lora_dropout: {self.config.lora_dropout}\n"
                    f"lora_target_modules: {self.config.lora_target_modules}\n"
                )
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.ddp = world_size != 1
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.config.gradient_accumulation_steps = (
                self.config.gradient_accumulation_steps // world_size
            )

        # use_wandb = len(wandb_project) > 0 or (
        #     "WANDB_PROJECT" in os.environ
        #     and len(os.environ["WANDB_PROJECT"]) > 0
        # )
        # # Only overwrite environ if wandb param passed
        # if len(wandb_project) > 0:
        #     os.environ["WANDB_PROJECT"] = wandb_project
        # if len(wandb_watch) > 0:
        #     os.environ["WANDB_WATCH"] = wandb_watch
        # if len(wandb_log_model) > 0:
        #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

        self.model_init()
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference

        if strategy in ["generation", "prompt"]:
            self.prompter = Prompter(prompt_template_name)

        self.data_loader(data_path)

    def model_init(
        self,
    ):
        if self.strategy not in ["sequence", "generation", "prompt"]:
            raise NotImplementedError(
                f"strategy type {self.strategy} not implemented"
            )

        if self.task_type not in ["binary", "regression"]:
            raise NotImplementedError(
                f"task type {self.task_type} not implemented"
            )

        if self.task_type == "binary" and self.strategy == "generation":
            raise NotImplementedError(
                "strategy generation cannot complete binary task yet."
            )

        model = None
        if self.strategy == "sequence":
            model = LlamaForSequenceClassification.from_pretrained(
                self.base_model,
                num_labels=1 if self.task_type == "regression" else 2,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.float16
                if self.task_type == "regression"
                else torch.int8,
                device_map=self.device_map,
            )
        elif self.strategy == "generation" or self.strategy == "prompt":
            model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )

        if model is None:
            raise RuntimeError("model init failed")

        # TODO int 8 may decrease the accuracy
        model = prepare_model_for_kbit_training(model)

        if self.peft:
            if self.peft_weights is not None:
                print("Peft weight file path:", self.peft_weights)
                model = PeftModel.from_pretrained(
                    model,
                    self.peft_weights,
                    torch_dtype=torch.float16,
                    is_trainable=True
                    if self.strategy == "sequence"
                    else False,
                )
            else:
                if self.strategy == "prompt":
                    with open(
                        os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "configs/prompt_design.json",
                        ),
                        "r",
                        encoding="utf8",
                    ) as rfile:
                        prompt_dict = json.loads(rfile.read())
                        prompt = prompt_dict[self.topic]
                    config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        num_virtual_tokens=30,
                        prompt_tuning_init_text=prompt,
                        tokenizer_name_or_path=self.base_model,
                    )
                elif self.strategy in ["generation", "sequence"]:
                    config = LoraConfig(
                        r=self.config.lora_r,
                        lora_alpha=self.config.lora_alpha,
                        target_modules=self.config.lora_target_modules,
                        lora_dropout=self.config.lora_dropout,
                        bias="none",
                        task_type=TaskType.SEQ_CLS
                        if self.strategy == "sequence"
                        else TaskType.CAUSAL_LM,
                    )
                model = get_peft_model(model, config)

            if self.resume_from_checkpoint:
                # Check the available weights and load them
                checkpoint_name = os.path.join(
                    self.resume_from_checkpoint, "pytorch_model.bin"
                )  # Full checkpoint
                if not os.path.exists(checkpoint_name):
                    checkpoint_name = os.path.join(
                        self.resume_from_checkpoint, "adapter_model.bin"
                    )  # only LoRA model - LoRA config above has to fit
                    self.resume_from_checkpoint = (
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

        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        model.config.use_cache = False
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

        self.model = model
        return model

    def generate_and_tokenize_prompt(self, data_point):
        text = sentence_cleaner(data_point["input"])
        inputs = self.prompter.generate_prompt(
            instruction=data_point["instruction"]
            if self.strategy == "generation"
            else None,
            input=text,
            label=None,
        )
        targets = data_point["output"]
        model_input = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.config.cutoff_len,
        )
        labels = self.tokenizer(targets)
        input_ids = model_input["input_ids"]
        label_input_ids = labels["input_ids"]
        if self.config.add_eos_token:
            label_input_ids += [self.tokenizer.eos_token_id]
        model_input["input_ids"] = input_ids + label_input_ids
        if not self.config.train_on_inputs:
            model_input["labels"] = [-100] * len(input_ids) + label_input_ids
        else:
            model_input["labels"] = input_ids + label_input_ids
        model_input["attention_mask"] = [1] * len(model_input["input_ids"])
        return model_input

    def sequence_tokenizer(self, data_point):
        text = sentence_cleaner(data_point["text"])

        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.config.cutoff_len
            and self.config.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = float(data_point["label"])

        return result

    def data_loader(self, data_path):
        if self.strategy == "sequence":
            dataset = load_dataset(
                "csv",
                data_files={
                    "train": f"{data_path}/train-{self.task_type}.csv",
                    "validate": f"{data_path}/validate-{self.task_type}.csv",
                    "test": f"{data_path}/test-{self.task_type}.csv",
                },
            )
            self.train_data = (
                dataset["train"].shuffle().map(self.sequence_tokenizer)
            )
            self.validate_data = (
                dataset["validate"].shuffle().map(self.sequence_tokenizer)
            )
            self.test_data = (
                dataset["test"].shuffle().map(self.sequence_tokenizer)
            )

            # signature_columns = ["input_ids", "attention_mask", "labels"]
            # ignored_columns = list(set(self.train_data.column_names) - set(signature_columns))

            self.train_data = self.train_data.remove_columns(["label"])
            self.validate_data = self.validate_data.remove_columns(["label"])
            self.test_data = self.test_data.remove_columns(["label"])

        elif self.strategy == "generation" or self.strategy == "prompt":
            dataset = load_dataset(
                "json",
                data_files={
                    "train": f"{data_path}/train.json",
                    "validate": f"{data_path}/val.json",
                    "test": f"{data_path}/test.json",
                },
            )
            self.train_data = (
                dataset["train"]
                .shuffle()
                .map(self.generate_and_tokenize_prompt)
            )
            self.validate_data = (
                dataset["validate"]
                .shuffle()
                .map(self.generate_and_tokenize_prompt)
            )
            self.test_data = (
                dataset["test"]
                .shuffle()
                .map(self.generate_and_tokenize_prompt)
            )
        print(
            f"\nFinish loading data: there are {len(self.train_data)} train data, {len(self.validate_data)} validation data, {len(self.test_data)} test data."
        )

    def binary_metrics_compute(self, pred):
        labels = pred.label_ids
        preds = (
            pred.predictions[0]
            if isinstance(pred.predictions, tuple)
            else pred.predictions
        )
        preds = np.argmax(preds, axis=1)
        # use when this is 0-1 classification task
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def regression_metrics_compute(self, pred):
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
            preds_set = self.error_analysis.get(x, np.array(0))
            preds_set = np.append(preds_set, preds[idx])
            self.error_analysis[x] = preds_set

        return {
            "rmse": rmse,
            "integerized_rmse": integerized_rmse,
            "label_precision_rmse": label_precision_rmse,
        }

    def default_hp_space_optuna(self, trial):
        return {
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0.01, 0.03, 0.05, 0.1, 0.2]
            ),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", [20, 40, 50, 60, 100, 200]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
            ),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs", 10, 20, log=True
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [16, 32, 64]
            ),
        }

    def train(self, parameter_search: bool = False):
        self.model.train()

        metric_for_best_model = ""
        if self.strategy == "sequence":
            metric_for_best_model = (
                "accuracy" if self.task_type == "binary" else "rmse"
            )
        else:
            metric_for_best_model = "loss"

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.validate_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.config.micro_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                fp16=True,
                logging_steps=self.config.logging_steps,
                optim=self.config.optim,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                output_dir=self.output_dir,
                save_total_limit=3,
                # remove_unused_columns=False,
                label_names=["labels"],
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                ddp_find_unused_parameters=False if self.ddp else None,
                group_by_length=False,
                # report_to="wandb",
                # run_name=wandb_run_name if use_wandb else None,
                report_to=["tensorboard"],
                logging_dir=self.log_dir,
            ),
            data_collator=transformers.DataCollatorWithPadding(
                self.tokenizer, return_tensors="pt"
            )
            if self.strategy == "sequence"
            else transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

        if self.strategy == "sequence":
            self.trainer.compute_metrics = (
                self.binary_metrics_compute
                if self.task_type == "binary"
                else self.regression_metrics_compute
            )

        # self.model.config.use_cache = False
        # old_state_dict = self.model.state_dict
        # self.model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(self.model, type(self.model))

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        if parameter_search:
            import optuna

            self.trainer.model_init = self.model_init
            best_run = self.trainer.hyperparameter_search(
                hp_space=lambda x: self.default_hp_space_optuna(x),
                backend="optuna",
                direction="minimize"
                if self.task_type == "regression"
                else "maxmize",
            )
            print("best_run", best_run)

            for n, v in best_run.hyperparameters.items():
                setattr(self.trainer.args, n, v)

            self.resume_from_checkpoint = False

        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.model.save_pretrained(self.output_dir)

        print("training finished!")

    def sequence_eval(self):
        self.model.eval()
        self.error_analysis = {}  # TODO can be optimized
        predictor = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                per_device_eval_batch_size=self.config.micro_batch_size,
                logging_steps=self.config.logging_steps,
                output_dir=self.output_dir,
                label_names=["labels"],
                # report_to="wandb",
                # run_name=wandb_run_name if use_wandb else None,
                logging_dir=self.log_dir,
                fp16=True,
                optim=self.config.optim,
                ddp_find_unused_parameters=False if self.ddp else None,
                group_by_length=False
                # remove_unused_columns=False,
            ),
            data_collator=transformers.DataCollatorWithPadding(
                self.tokenizer, return_tensors="pt"
            )
            if self.strategy == "sequence"
            else transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
            compute_metrics=self.regression_metrics_compute
            if self.task_type == "regression"
            else self.binary_metrics_compute,
        )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        # TODO
        pred = predictor.predict(test_dataset=self.test_data)
        print(pred.metrics)
        prediction = pred.predictions[0].flatten()
        prediction = np.clip(prediction, -2, 2)

        rmse_dict = {}
        for k, v in pred.metrics.items():
            print(f"{k}:    {v}")
        for k, s in self.error_analysis.items():
            true = np.ones(s.shape) * k
            rmse = mean_squared_error(true, s, squared=False)
            rmse_dict[k] = rmse
            # true = torch.from_numpy(true)
            # s = torch.from_numpy(s)
            # huber = creterion(s, true)
            print(f"{k}:    rmse-{rmse}")
            # log(log_file, f'{str(k)}:    rmse-{str(rmse)}; huber-{huber}')

        torch.cuda.empty_cache()

    def single_prompt_evaluate(
        self,
        prompt="",
        temperature=0.4,
        top_p=0.65,
        top_k=35,
        repetition_penalty=1.1,
        max_new_tokens=20,
        **kwargs,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        return self.prompter.get_response(output)

    def generation_eval(self):
        self.model.eval()
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "configs/label_map.json",
            ),
            "r",
            encoding="utf8",
        ) as rfile:
            translator_dict = json.loads(rfile.read())
            translator = translator_dict[self.topic]

        prediction = np.array([])
        true = np.array([])
        print("evaluate begin...")
        for single_test in tqdm(self.test_data):
            single_prompt = self.prompter.generate_prompt(
                instruction=single_test["instruction"]
                if self.strategy == "generation"
                else None,
                input=single_test["input"],
            )
            result = self.single_prompt_evaluate(single_prompt)
            output = result_translator(self.topic, result, translator)
            prediction = np.append(prediction, output)
            label = result_translator(
                self.topic, single_test["output"], translator
            )
            true = np.append(true, label)
            print(
                f"result: {single_prompt}{result}\noutput: {output}\nlabel: {label}\n"
            )

        acc = accuracy_score(true, prediction)
        data = pd.DataFrame(data={"predict": prediction, "true": true})
        irrelevant_eval = data.replace({2: 1, 1: 1, 0: 1, -1: 1, -2: 1, -9: 0})
        relevant_data = data.drop(
            data[(data["true"] == -9) | (data["predict"] == -9)].index
        )

        relevant_acc = accuracy_score(
            irrelevant_eval.true.values, irrelevant_eval.predict.values
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            irrelevant_eval.true.values,
            irrelevant_eval.predict.values,
            average="binary",
        )

        rmse = mean_squared_error(
            relevant_data.true.values,
            relevant_data.predict.values,
            squared=False,
        )
        for _, row in relevant_data.iterrows():
            preds_set = self.error_analysis.get(row["true"], np.array(0))
            preds_set = np.append(preds_set, row["predict"])
            self.error_analysis[row["true"]] = preds_set

        print(f"total acc: {acc}\n")
        print(
            f"ir/relevant: acc-{relevant_acc}, precision-{precision}, recall-{recall}, f1-{f1}\n"
        )
        print(f"rmse: {rmse}\n")
        print("error_analysis: \n")

        for k, s in self.error_analysis.items():
            true = np.ones(s.shape) * k
            rmse = mean_squared_error(true, s, squared=False)

            print(f"{str(k)}:    {str(rmse)}")

        return acc, rmse

    def eval(self):
        if self.strategy == "sequence":
            self.sequence_eval()
        else:
            self.generation_eval()

    def predict(
        self,
        texts: list,
        max_length: int = 128,
        batch: int = 64,
        verbose=print,
    ) -> np.array:
        verbose(
            f"predict(texts={len(texts)}, max_length={max_length}, batch={batch})"
        )
        self.model.eval()
        torch.cuda.empty_cache()
        try:
            # verbose(f'self.task_type={self.task_type}')
            if self.task_type in [
                "regression",
                "binary",
            ]:  # output one score for each input
                prediction = np.array([])
            verbose(f"initial prediction.shape={prediction.shape}")

            with torch.no_grad():
                for i in range(0, len(texts), batch):
                    # encode input texts
                    encoding = self.tokenizer(
                        [
                            sentence_cleaner(single_text)
                            for single_text in texts[i : i + batch]
                        ],
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                        return_tensors="pt",
                    )
                    if torch.cuda.is_available():
                        # verbose(f'self.model.device={self.model.device}')  # self.model.device=cuda:0
                        for key in encoding.keys():
                            encoding[key] = encoding[key].cuda()
                            # verbose(f'encoding[{key}].device={encoding[key].device}. encoding[{key}].shape={encoding[key].shape}') # encoding[input_ids].device=cuda:0. encoding[input_ids].shape=torch.Size([4, 128])

                    # calculate the encoded input with frozen model
                    outputs = self.model(
                        input_ids=encoding["input_ids"]
                    ).logits.detach()
                    # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}') #  type(outputs)=<class 'torch.Tensor'>, outputs.device=cuda:0
                    if torch.cuda.is_available():
                        outputs = (
                            outputs.cpu()
                        )  # copy the tensor to host memory before converting it to numpy. otherwise we will get an error "can't convert cuda:0 device type tensor to numpy"
                        # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}')  # type(outputs)=<class 'torch.Tensor'>, outputs.device=cpu
                    del encoding
                    verbose(f"outputs.numpy().shape={outputs.numpy().shape}")
                    # verbose(f'outputs.numpy()={outputs.numpy()}')

                    # output
                    if "regression" == self.task_type:
                        prediction = np.append(
                            prediction, outputs.numpy().flatten()
                        )  # a score for each input string
                    elif "binary" == self.task_type:
                        prediction = np.append(
                            prediction, special.expit(outputs.numpy()[:, 1])
                        )  # a float probability of label "1" for each input string
                    elif self.task_type in ["climate", "ternary"]:
                        # verbose(f'append outputs.numpy() as x3 arrays.')
                        prediction = np.append(
                            prediction, outputs.numpy(), axis=0
                        )  # three probabilities for each input string. outputs.shape=(num of inputs, 3)
                        # verbose(f'prediction={prediction}')
                    else:
                        raise ValueError(
                            f"Unknown self.task_type = {self.task_type}."
                        )
                    del outputs
                    torch.cuda.empty_cache()
                    verbose(f"prediction.shape={prediction.shape}")
                    # verbose(f'prediction={prediction}')

                # output
                return prediction

        except RuntimeError as error:
            verbose(f"Running out of memory, retrying with a smaller batch.")
            # if ('CUDA out of memory' in str(error)) and (batch >= 2):
            #     batch = int(batch / 2)
            #     verbose(f'Device name = {torch.cuda.get_device_name(0)}. torch.cuda.mem_get_info() = free {torch.cuda.mem_get_info(0)[0] / 1000000000:.1f} GB , total {torch.cuda.mem_get_info(0)[1] / 1000000000:.1f} GB')
            #     if 'encoding' in locals():
            #         del encoding
            #     if 'outputs' in locals():
            #         del outputs
            #     for parameter in self.model.parameters():
            #         parameter.requires_grad = False  # freeze the finetuned model. save memory.
            #         if parameter.grad is not None:
            #             del parameter.grad  # save memory.
            #     torch.cuda.empty_cache()
            #     verbose(f'Device name = {torch.cuda.get_device_name(0)}. torch.cuda.mem_get_info() = free {torch.cuda.mem_get_info(0)[0] / 1000000000:.1f} GB , total {torch.cuda.mem_get_info(0)[1] / 1000000000:.1f} GB')
            #     return self.predict(texts=texts, max_length=max_length, padding=padding, batch=batch)
            # else:
            raise RuntimeError(
                "Running out of GPU memory. Try limiting [max_length] and [batch]."
            ) from error

    # predict()
