from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper, HuggingfaceEmbeddings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


langchain_embeddings = HuggingfaceEmbeddings(model="OrdalieTech/Solon-embeddings-large-0.1")
# langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
loader = DirectoryLoader("your-directory")
documents = loader.load()

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_KEY"))
critic_llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_KEY"))
embeddings = HuggingfaceEmbeddings

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# adapt to language
language = "français"

generator.adapt(language, evolutions=[simple, reasoning, conditional, multi_context])
generator.save(evolutions=[simple, reasoning, multi_context, conditional])

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
