import os

from dotenv import find_dotenv, load_dotenv
from unsloth import FastLanguageModel

import wandb

load_dotenv(find_dotenv())

run = wandb.init()
model = run.use_model("adrienb134/model-registry/croissant-asn-rag:production")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

model.save_pretrained_merged(
    "asn_rag_prod",
    tokenizer,
    save_method="merged_16bit",
)
model.push_to_hub_merged("AdrienB134/asn_rag_prod", tokenizer, save_method="merged_16bit", token=os.getenv("HF_KEY"))
