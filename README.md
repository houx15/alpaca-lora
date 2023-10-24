# LLM (Large Language Model) for Tweet Opinion Analysis

This repository contains code for training models to analyze a tweet's opinion on several topics (gun control, abortion, china favorability, sexual orientation, climate change). The analyse is based on [meta's llama2 models](https://huggingface.co/meta-llama). 

To fine-tune cheaply and efficiently, we use [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf) with Hugging Face's [PEFT](https://github.com/huggingface/peft) as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

## Usage

### Local Setup

1. Install dependencies

   To get the full environment, run:
   ```bash
   pip install -r requirements.txt
   ```

   Users who only need to run the inference part, run:
   ```bash
   conda env create -f predict.yaml
   ```
   to install the minimum environment.

2. Cache base models [Optional]

   Hugging Face downloads models automatically from their online hub. However, you may get an error if your environment cannot access to the Internet. So we recommend you to cache the models manually before training or prediction.

   ```bash
   python download_models.py --model "your-model-id"
   ```

   The model id can be accessed on [Hugging Face](https://huggingface.co/models). For example, the model id of [llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf) is "meta-llama/Llama-2-13b-hf". And you can cache it by running:

   ```bash
   python download_models.py --model "meta-llama/Llama-2-13b-hf"
   ```

   !NOTE: Meta Llama2 model on Hugging Face is a gated model, and you need to apply for the access first on [this page](https://huggingface.co/meta-llama/Llama-2-13b-hf/tree/main).

   After your application is granted (within 1 day), you need to first login your Hugging Face account through access tokens on your server:

   ```bash
   huggingface-cli login
   ```
    
   The access token can be created on [this page](https://huggingface.co/settings/tokens).

   Then you can successfully cache the llama2 models:

   ```bash
   python download_models.py --model "meta-llama/Llama-2-13b-hf"
   ```

### Training & Evaluation (`model.py`)

This file contains a straightforward application of PEFT to the LLaMA model. We also provide a simple script `run.py` for users to finetune or evaluate the model easily.

Example usage:

```bash
python run.py \
    --topic 'abortion' 
    --do_train 
    --model_type 'llama-2-13b' 
    --task_type 'regression'
```

We can also tweak our strategies:

```bash
python run.py \
    --topic 'abortion' #abortion / gun / china / sexual / climate / drug
    --do_train  #do_train or do_eval
    --model_type 'llama-2-13b' #llama-2-7b or llama-2-13b
    --task_type 'regression' #binary or regression
    --dataset_update #regenerate the dataset based on the source data
    --augmentation #perform data augmentation or not
    --strategy #sequence: sequence classification; prompt: prompt tuning
    --peft #whether use the peft strategy or not
    --use_pretrained_peft_weights
    --eval_model_path './output' #the adapter weights for evaluation
    --parameter_search #use optuna to search for the best parameters
    --output_dir_base # specify a base to put outputs
    --output_dir #f"{output_dir_base}output/{strategy}/{model_type}/{topic}/{task_type}" by default
    --log_dir # f"logs/{strategy}/{model_type}/{topic}/{task_type}" by default
```

The hyperparameters are saved in the `train_para/` folder, you may tweak the parameters there.

### Inference (`predict.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from the specified `model_path`, and runs a sequence classification task to output the irrelevance or opinion of a particular tweet.

Example usage:

```python
from .predict import OpinionPredict
predictor = OpinionPredict(
    task_type="regression", #"regression" or "binary"
    model_path="path/to/the/saved/adapter/weights",
    model_base="llama-2-13b", #[Optional], llama-2-13b by default
)

import pandas as pd

data = pd.read_csv("the/data/file.csv")
texts = data["text"].values # a numpy-array (n*1) of texts

result = predictor.predict(texts=texts, batch=16) # a numpy-array (n*1) of float numbers.
```


## Performance

### 1. Sequence Classification

| Topic | BERT-Binary | BERT-Regression | Llama2-13b-Binary | Llama2-13b-Regression | Alpaca-7b-Binary | Alpaca-7b-Regression | Alpaca-13b-Binary | Alpaca-13b-Regression |
|---|---|---|---|---|---|---|---|---|
|Gun Control|88%|0.97|87%|0.98|||||
|Climate Change||1.01|94.73%|0.86|||||
|China favorability|89%|0.85|85.6%|0.82|||||
|Abortion|92%|0.98|95%|0.90|||96.71%|0.88|
|Sexual orientation|92%|0.69|92%||||||

### 2. Fine Tuning With Hard Prompt

| Topic | Llama2-13b-chat-Binary | Llama2-13b-chat-Regression |
| --- | --- | --- |
| Gun Control | 92% | 0.745 |
| Climate Change | 100% | 0.54 |
| China Favoribility | 81.4% | 0.706 |
| Abortion | 97.4% | 0.612 |
| Sexual Orientation | 96.4% | 0.724 |

### 3. Prompt Tuning

| Topic | Llama2-13b-chat-Regression |
| --- | --- |
| Gun Control | 1.217 |
| Climate Change | 1.43 |
| China Favoribility | 0.837 |
| Abortion | 0.628 | 
| Sexual Orientation | 0.808 |

## Technique Details

### Data Description

#### 1. Regression
|Topic | Labeller | Cohen's Kappa | Other |
|---|---|---|---|
|Gun Control||||
|Climate Change||||
|China favorability||||
|Abortion||||
|Sexual orientation||||

#### 2. Binary
|Topic | Labeller | Cohen's Kappa | Other |
|---|---|---|---|
|Gun Control||||
|Climate Change||||
|China favorability||||
|Abortion||||
|Sexual orientation||||

### Work Flow
