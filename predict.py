import torch
import numpy as np

from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
)
from transformers.utils import PaddingStrategy

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)

from scipy import special

from typing import Union
from utils.utils import sentence_cleaner


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
        self, task_type: str, model_path: str, model_base: str = "llama-2-13b"
    ) -> None:
        try:
            assert task_type in ["regression", "binary"]
        except:
            raise AssertionError(f"task type {task_type} is not supported")

        try:
            assert model_base in model_dict.keys()
        except:
            raise AssertionError(f"model base {model_base} is not supported")
        self.task_type = task_type
        model_base = model_dict[model_base]

        self.tokenizer = LlamaTokenizer.from_pretrained(model_base)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        self.model = self.model_init(model_base, model_path)

    def model_init(self, model_base, model_path):
        torch.manual_seed(42)
        model = LlamaForSequenceClassification.from_pretrained(
            model_base,
            num_labels=1 if self.task_type == "regression" else 2,
            load_in_8bit=True,
            torch_dtype=torch.float16
            if self.task_type == "regression"
            else torch.int8,
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
        verbose(
            f"predict(texts={len(texts)}), max_length={max_length}, batch={batch}"
        )
        self.model.eval()

        prediction = np.array([])

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

        return prediction


if __name__ == "__main__":
    a = OpinionPredict(
        task_type="regression",
        model_path="/scratch/network/yh6580/output/sequence/llama-2-13b/abortion/regression/saved0902",
    )
    import pandas as pd

    data = pd.read_csv("dataset/abortion/test-regression.csv")
    texts = data["text"].values

    result = a.predict(texts=texts, batch=32)
    print(result)
