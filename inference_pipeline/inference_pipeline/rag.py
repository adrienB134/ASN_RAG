import os
import time
from typing import Tuple

import dspy
import wandb
from dotenv import find_dotenv, load_dotenv
from dsp import LM
from dspy.predict import Retry
from dspy.retrieve.chromadb_rm import ChromadbRM

from ragatouille import RAGPretrainedModel
from utils.embeddings import HuggingFaceEmbeddingFunction

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


class MultiQueryRAG(dspy.Module):
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
            context = context + passages
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
    def __init__(self, lm: LM, final_writer: LM) -> None:
        self.lm = lm
        self.final_writer = final_writer
        self.lm.drop_prompt_from_output = True
        self.rm = ChromadbRM(
            collection_name="ASN_test",
            persist_directory="../data",
            embedding_function=huggingface_ef,
            k=20,
        )
        dspy.settings.configure(lm=self.lm, rm=self.rm)
        self.reranker = RAGPretrainedModel.from_pretrained("AdrienB134/ColBERTv2.0-mmarcoFR")
        self.pipe = MultiQueryRAG(reranker=self.reranker, max_hops=2, final_writer=self.final_writer)

    def query(self, question) -> Tuple[str, list]:
        answer, sources = self.pipe.forward(question)
        print(f"Réponse: {answer.reponse}")
        return answer.reponse, sources

    # * For testing purposes
    def _retrieve(self, question):
        return self.rm.forward(question, k=7)

    def _rerank(self, question, passages):
        return self.reranker.rerank(question, passages, k=7)


# croissantllm/CroissantLLMChat-v0.1
