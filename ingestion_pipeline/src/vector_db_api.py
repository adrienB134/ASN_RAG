import chromadb
from chromadb.api import ClientAPI

from src.paths import DATA_DIR


def get_chroma_client() -> ClientAPI:
    chroma_client = chromadb.PersistentClient(path=str(DATA_DIR))
    return chroma_client


def get_or_create_chromadb_collection(
    chroma_client: ClientAPI,
    collection_name: str,
):
    return chroma_client.get_or_create_collection(name=collection_name)
