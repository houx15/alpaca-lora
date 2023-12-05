import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    GenerationConfig,
)
from transformers.utils import PaddingStrategy

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)

from scipy import special

from typing import Union
from utils.utils import sentence_cleaner, result_translator
from utils.prompter import Prompter


model_dict = {
    "7b": "chainyo/alpaca-lora-7b",
    "13b": "yahma/llama-13b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}


class OpinionPredict(object):
    r"""
    Make predictions based on trained adapters.

    Args:
        task_type (`str`):
            Current task type, 'regression' or 'binary'
        model_path (`str`):
            The path of trained adapters.
        model_base (`str`, *optional*, defaults to 'llama2-13b'):
            Base LLM
    """

    def __init__(
        self, task_type: str, model_path: str, strategy: str = "sequence", model_base: str = "llama-2-13b", topic: str = "None"
    ) -> None:
        try:
            assert task_type in ["regression", "binary"]
        except:
            raise AssertionError(f"task type {task_type} is not supported")

        try:
            assert strategy in ["sequence", "generation", "prompt"]
        except:
            raise AssertionError(f"strategy {strategy} is not supported")
    
        try:
            assert model_base in model_dict.keys()
        except:
            raise AssertionError(f"model base {model_base} is not supported")

        self.task_type = task_type
        self.strategy = strategy
        self.topic = topic
        model_base = model_dict[model_base]

        self.tokenizer = LlamaTokenizer.from_pretrained(model_base)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        self.model = self.model_init(model_base, model_path)
        
        if self.strategy == "generation" or self.strategy == "prompt":
            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "configs/label_map.json",
                ),
                "r",
                encoding="utf8",
            ) as rfile:
                    translator_dict = json.loads(rfile.read())
                    try:
                        self.translator = translator_dict[topic]
                    except:
                        raise KeyError(f"topic {topic} is not included in translator.")
            
            prompter_template = "alpaca" if self.strategy == "generation" else "prompt_tuning"
            print(f"Using template {prompter_template} for the prompter")
            self.prompter = Prompter(prompter_template)
        
        if self.strategy == "generation":
            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "configs/prompt_design.json",
                ),
                "r",
                encoding="utf8",
            ) as rfile:
                prompt_dict = json.loads(rfile.read())
                try:
                    self.prompt = prompt_dict[topic]
                except:
                    raise KeyError(f"topic {topic} is not included in the prompt dict.")

            

    def model_init(self, model_base, model_path):
        torch.manual_seed(42)
        model = None
        if self.strategy == "sequence":
            model = LlamaForSequenceClassification.from_pretrained(
                model_base,
                num_labels=1 if self.task_type == "regression" else 2,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif self.strategy == "generation" or self.strategy == "prompt":
            model = LlamaForCausalLM.from_pretrained(
                model_base,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        model = prepare_model_for_kbit_training(model)

        print(f"loading peft weights: {model_path}")
        model = PeftModel.from_pretrained(
            model, model_path, torch_dtype=torch.float16, is_trainable=False
        )
        model.print_trainable_parameters()
        return model

    def predict(
        self,
        texts: list,
        max_length: int = 256,
        padding: Union[bool, str, PaddingStrategy] = "longest",
        batch: int = 64,
        verbose=print,
    ) -> np.array:
        """
        :param texts: list (or numpy.array, pandas.Series, torch.tensor) of strings.
        :param padding: str. 'longest' (default), 'max_length', 'do_not_pad'.
        :param max_length, batch: int. max length of tweet and number of tweets proceeded in a batch. limited by GPU memory.
            max_length=128 is sufficient for most tweets, and 512 tweeets per batch are recommended for 128-letter tweets on a typical Tesla GPU with 16GB memory.
            :param verbose: function that accepts string input. used to display message.
            e.g., verbose = print (default, display on screen)
            e.g., verbose = logger.info (write to a log file)
            e.g., verbose = lambda message: logger.info(f'filename: {message}') # write to a log file with a header
        :return: 1-dim numpy array
        :example:
            >>> from predict import OpinionPredict
            >>> OpinionPredict(task_type='regression', model_path='../model-llama2/2023-09-04-yuxin-llama2-abortion-regression-090'
            ).predict(texts=[
                "RT After the shooting, Jim and his wife Sarah dedicated their lives to preventing gun violence. They were lifelong Republicans and gun owners themselves. They realized that passing sensible gun laws isn't about politics; it's about saving lives. #GunReform ",
                'Did u hear the gunshot in the video, when someone is rushing you and u hear a gunshot, they could have a gun so u shoot them. Someone else fired a gun.',
                "It didn't though. You can literally 3d print a gun now anyways, no use in banning them. Also that's a one way ticket to all out civil war",
                'I repeat, the 2nd amendment is not on trial here. Kyle did not engage, others engaged him. You are allowed to eliminate as many threats as is necessary to preserve your own life. There is no limit after which the right to your own life becomes inferior to that of your attackers.'
            ])

            output: array([ 1.7457896 ,  1.0045699 ,  0.07550862, -0.76812345], dtype=float32)
            ground truth: [2, 1, 0, -1]
        """
        verbose(
            f"predict(texts={len(texts)}), max_length={max_length}, batch={batch}"
        )
        self.model.eval()

        prediction = np.array([])
        
        if self.strategy == "sequence":
            for i in range(0, len(texts), batch):
                encoding = self.tokenizer(
                    [
                        sentence_cleaner(single_text)
                        for single_text in texts[i : i + batch]
                    ],
                    truncation=True,
                    max_length=max_length,
                    padding=padding,
                    return_tensors="pt",
                )
                if torch.cuda.is_available():
                    for key in encoding.keys():
                        encoding[key] = encoding[key].cuda()

                outputs = self.model(**encoding).logits.detach()

                if torch.cuda.is_available():
                    outputs = outputs.cpu()

                verbose(f"outputs.numpy().shape={outputs.numpy().shape}")
                # output
                if "regression" == self.task_type:
                    prediction = np.append(
                        prediction, outputs.numpy().flatten()
                    )  # a score for each input string
                elif "binary" == self.task_type:
                    prediction = np.append(
                        prediction, special.expit(outputs.numpy()[:, 1])
                    )  # a float probability of label "1" for each input string
                verbose(f"prediction.shape={prediction.shape}")

        else:
            texts_with_prompt = [self.prompter.generate_prompt(
                instruction=self.prompt if self.strategy == "generation" else None,
                input=text,
            ) for text in texts]
            
            max_new_tokens = 20
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=max_new_tokens,
            )
            
            for text_with_prompt in tqdm(texts_with_prompt):
                encoding = self.tokenizer(text_with_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    encoding["input_ids"] = encoding["input_ids"].cuda()
                generate_params = {
                    "input_ids": encoding["input_ids"],
                    "generation_config": generation_config,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "max_new_tokens": max_new_tokens,
                }
                with torch.no_grad():
                    generation_output = self.model.generate(
                        input_ids=encoding["input_ids"],
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                    )
                # if torch.cuda.is_available():
                #     generation_output = generation_output.cpu()
                output = self.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
                result = self.prompter.get_response(output)
                prediction = np.append(prediction, int(result_translator(self.topic, result, self.translator)))
            
        return prediction


if __name__ == "__main__":
    
    predictor = OpinionPredict(
        task_type="regression",
        strategy="generation",
        model_base="llama-2-13b-chat",
        # model_path="/scratch/network/dg2944/alpaca-lora/output/generation/llama-2-13b-chat/gun/regression",
        topic="gun",
    )

    data = pd.read_csv("dataset/gun/test-regression.csv")
    texts = data["text"].values

    result = predictor.predict(texts=texts, batch=8)
    print(result)