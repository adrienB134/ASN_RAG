import os
import time
from typing import Tuple

import dspy
import wandb
from dotenv import find_dotenv, load_dotenv
from dsp.modules import Claude

# from dsp.utils import deduplicate
from dspy.predict import Retry
from dspy.retrieve.chromadb_rm import ChromadbRM
from ragatouille import RAGPretrainedModel
from utils.embeddings import HuggingFaceEmbeddingFunction
from utils.hf_client_tgi import HFClientTGI
from utils.unsloth_model import UnslothModel

load_dotenv(find_dotenv())

huggingface_ef = HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HF_KEY"),
    model_name="OrdalieTech/Solon-embeddings-large-0.1",
)


def deduplicate(context: list) -> list:
    dedup = set()
    for i in context:
        dedup.add(i.id)
    new_list = []
    for i in context:
        if i.id in dedup:
            new_list.append(i)
            dedup.remove(i.id)
    return new_list


class GenerateSearchQuery(dspy.Signature):
    """Ecrit une simple question qui aidera a répondre a une question complexe"""

    # """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="contient peut-être des informations utiles")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateCitedParagraph(dspy.Signature):
    """Génère un paragraphe avec des citations"""

    # """Generate a paragraph with citations."""

    context = dspy.InputField(desc="contient peut-être des informations utiles")
    question = dspy.InputField()
    reponse = dspy.OutputField(desc="inclus citations et liste les sources")


class BasicRAG(dspy.Module):
    def __init__(self, reranker: RAGPretrainedModel, max_hops: int, final_writer) -> None:
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=7)
        self.reranker = reranker
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
        self.final_writer = final_writer

    def forward(self, question: str) -> Tuple[dspy.Prediction, list]:
        timer = time.time()
        print("retrieving!")
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](
                context=[c["long_text"] for c in context],
                question=question,
            ).query
            print(query)

            passages = dspy.settings.rm.forward(
                f"query: {query}"  # See https://huggingface.co/OrdalieTech/Solon-embeddings-large-0.1
            )
            context = context + passages  # TODO Deduplicate
        context = deduplicate(context)
        print(f"\nRetrieval took {time.time() - timer} s\n")

        timer = time.time()
        print("reranking!")
        reranked_context = self.reranker.rerank(query, [a["long_text"] for a in context], k=10)

        for a, b in zip(reranked_context, [a["metadatas"]["Source"] for a in context]):
            a["source"] = b

        print(f"\nReranking took {time.time() - timer} s\n")

        timer = time.time()
        with dspy.context(lm=self.final_writer):
            pred = self.generate_cited_paragraph(context=context, question=question)
        print(f"\nGeneration took {time.time() - timer} s\n")
        answer = dspy.Prediction(contexte=context, reponse=pred.reponse)
        return answer, reranked_context


class RAG:
    def __init__(self) -> None:
        # run = wandb.init()
        # model = run.use_model("adrienb134/model-registry/croissant-asn-rag:production")

        # self.lm = UnslothModel(model=artifact_dir)
        # self.lm.kwargs["max_tokens"] = 300
        # self.lm.kwargs["temperature"] = 0.2
        # self.lm.kwargs["top_p"] = 0.95
        # self.lm.kwargs["top_k"] = 60

        self.lm = dspy.OpenAI(
            model="gpt-3.5-turbo",
            max_tokens=3800,
            api_key=os.getenv("OPENAI_KEY"),
        )
        self.gpt4 = dspy.OpenAI(
            model="gpt-4-0125-preview",
            max_tokens=3800,
            api_key=os.getenv("OPENAI_KEY"),
        )
        # self.lm = HFClientTGI(
        #     model="occiglot/occiglot-7b-fr-en-instruct",
        #     url="https://cbljfyw02b3p7c2u.us-east-1.aws.endpoints.huggingface.cloud",
        #     port=8080,
        #     token=os.getenv("HF_KEY"),
        # )

        # self.lm = dspy.Cohere(model="command-r-plus", max_tokens=4000, api_key=os.getenv("COHERE"))
        # self.lm = Claude(
        #     model="claude-3-opus-20240229",
        #     api_key=os.getenv("CLAUDE"),
        #     max_tokens=4096,
        # )
        self.lm.drop_prompt_from_output = True
        self.rm = ChromadbRM(
            collection_name="ASN_test",
            persist_directory="../data",
            embedding_function=huggingface_ef,
            k=20,
        )
        dspy.settings.configure(lm=self.lm, rm=self.rm)
        self.reranker = RAGPretrainedModel.from_pretrained("antoinelouis/colbertv2-camembert-L4-mmarcoFR")
        self.pipe = BasicRAG(reranker=self.reranker, max_hops=2, final_writer=self.gpt4)

    def query(self, question) -> None:
        answer, sources = self.pipe.forward(question)
        print(f"Réponse: {answer.reponse}")
        return answer.reponse, sources

    def _retrieve(self, question):
        return self.rm.forward(question, k=7)

    def _rerank(self, question, passages):
        return self.reranker.rerank(question, passages, k=7)


# croissantllm/CroissantLLMChat-v0.1
