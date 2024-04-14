import os

import dspy
import wandb
from dotenv import find_dotenv, load_dotenv
from dsp.modules import Claude
from utils.hf_client_tgi import HFClientTGI
from utils.unsloth_model import UnslothModel

load_dotenv(find_dotenv())

# wandb_model = UnslothModel(model=wandb.use_model("adrienb134/model-registry/croissant-asn-rag:production"))
# wandb_model.kwargs["max_tokens"] = 300
# wandb_model.kwargs["temperature"] = 0.2
# wandb_model.kwargs["top_p"] = 0.95
# wandb_model.kwargs["top_k"] = 60

ollama_lm = dspy.OllamaLocal(
    model="command-r",
    base_url="http://localhost:11434",
    temperature=0.2,
    top_p=0.95,
    max_tokens=2000,
)

gpt_3_5_turbo = dspy.OpenAI(
    model="gpt-3.5-turbo",
    max_tokens=3800,
    api_key=os.getenv("OPENAI_KEY"),
)


gpt_4_turbo = dspy.OpenAI(
    model="gpt-4-0125-preview",
    max_tokens=3800,
    api_key=os.getenv("OPENAI_KEY"),
)


occiglot_tgi = HFClientTGI(
    model="occiglot/occiglot-7b-fr-en-instruct",
    url="https://cbljfyw02b3p7c2u.us-east-1.aws.endpoints.huggingface.cloud",
    port=8080,
    token=os.getenv("HF_KEY"),
)


cohere = dspy.Cohere(model="command-r-plus", max_tokens=4000, api_key=os.getenv("COHERE"))


claude = Claude(
    model="claude-3-opus-20240229",
    api_key=os.getenv("CLAUDE"),
    max_tokens=4096,
)
