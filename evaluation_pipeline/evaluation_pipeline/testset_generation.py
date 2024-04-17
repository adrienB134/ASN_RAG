import os

from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI
from ragas.embeddings import HuggingfaceEmbeddings, LangchainEmbeddingsWrapper
from ragas.testset.evolutions import conditional, multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator
from tqdm import tqdm

load_dotenv(find_dotenv())

openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_KEY"))
langchain_embeddings = HuggingfaceEmbeddings(model="OrdalieTech/Solon-embeddings-large-0.1")
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
loader = DirectoryLoader("/home/adrien/Documents/Coding/RAG_ASN/Ingestion/ASN/txt")
documents = loader.load()


for document in tqdm(documents):
    document.metadata["filename"] = document.metadata["source"]


# generator with openai models
generator_llm = openai
critic_llm = openai
embeddings = langchain_embeddings

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# adapt to language
language = "french"

generator.adapt(language, evolutions=[simple, reasoning, conditional, multi_context], cache_dir="./cache/french")
generator.save(evolutions=[simple, reasoning, multi_context, conditional], cache_dir="./cache")

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions={
        simple: 0.4,
        reasoning: 0.2,
        multi_context: 0.2,
        conditional: 0.2,
    },
    raise_exceptions=False,
    is_async=False,
)

testset = testset.to_dataset()

testset.push_to_hub(repo_id="AdrienB134/testset_asn_rag", token=os.environ["HF_KEY"])
