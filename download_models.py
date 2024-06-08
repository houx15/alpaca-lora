from transformers import pipeline, AutoTokenizer
from transformers.utils.hub import hf_cache_home

import argparse

# Lets do a quick check to see if the hugging face cache home is set
# to something with scratch in the path (\scratch\network\ on adroit,
# \scratch\gpfs on della-gpu).

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None)

config = parser.parse_args()

if config.model is None:
    raise ValueError("You must specify a model name")

if "/scratch" not in hf_cache_home:
    raise ValueError(
        f"Your hugging face home directory is set to '{hf_cache_home}'! "
        f"You will run out of space on della-gpu or adroit quickly "
        f"because cached models and datasets are large. Set the environment "
        f"variable HF_HOME to /scratch/network/<NetID>/.cache/huggingface "
        f"adroit or /scratch/gpfs/<NetID>/.cache/huggingface on della-gpu."
    )


# Simply instatiating pipelines will trigger downloading the models
# The specification of models can be left off, however, in production
# this is not a good practice as the default models can change and
# unexpected behaviour can occur. It is also best to specify a model
# revision has well.
tokenizer = AutoTokenizer.from_pretrained(config.model)
llm_pipeline = pipeline(task="ner", model=config.model)
