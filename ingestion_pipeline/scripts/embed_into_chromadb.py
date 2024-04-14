import hashlib
import multiprocessing
import os
from multiprocessing import set_start_method
from typing import List, Optional

import torch
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoModel, AutoTokenizer
from unstructured.cleaners.core import clean
from unstructured.cleaners.extract import extract_text_after
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.huggingface import chunk_by_attention_window

from src.logger import get_console_logger
from src.paths import DATA_DIR, FOLDER_DIR
from src.vector_db_api import get_chroma_client, get_or_create_chromadb_collection

COLLECTION_NAME = "ASN_test"

logger = get_console_logger()

tokenizer = AutoTokenizer.from_pretrained("OrdalieTech/Solon-embeddings-large-0.1")
model = AutoModel.from_pretrained("OrdalieTech/Solon-embeddings-large-0.1")

chromadb = get_chroma_client()
chromadb = get_or_create_chromadb_collection(chromadb, collection_name=COLLECTION_NAME)


class Document(BaseModel):
    id: str
    metadata: Optional[List[dict]] = []
    text: Optional[str] = ""
    chunks: Optional[list] = []
    chunk_ids: Optional[list] = []
    embeddings: Optional[list] = []


def parse_and_clean_document(_file: str) -> Document:
    document_id = hashlib.md5(_file.encode()).hexdigest()
    document = Document(id=document_id)
    document.metadata.append({"Source": f"{_file}".strip(".pdf")})

    logger.log(level=0, msg="Parsing document")
    loader = partition_pdf(filename=f"{FOLDER_DIR}/{_file}", strategy="ocr_only", languages=["fra"])
    document.text = "\n".join([x.text for x in loader])

    logger.log(level=0, msg="Cleaning document")
    document.text = clean(document.text, extra_whitespace=True)

    with open(f"{DATA_DIR}/txt/{_file.split('.')[0]}.txt", "w") as f:
        f.write(document.text)

    # try:
    #     document.text = extract_text_after(document.text, r"(?i)OBJET\s?:")
    # except:
    #     print(f"{_file} not cleaned")

    return document


def chunk(document: Document) -> Document:
    chunks = []
    chunks += chunk_by_attention_window(document.text, tokenizer)
    document.chunks = chunks
    chunk_ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in document.chunks]
    document.chunk_ids.append(chunk_ids)

    document.metadata = [document.metadata[0] for _ in document.chunks]
    return document


def embedding(document: Document) -> Document:
    for chunk in document.chunks:
        inputs = tokenizer(
            chunk,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        result = model(**inputs)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        lst = embeddings.flatten().tolist()
        document.embeddings.append(lst)

    return document


def add_document_to_collection(document: Document, collection=chromadb) -> None:
    collection.add(
        ids=document.chunk_ids[0],
        embeddings=document.embeddings,
        metadatas=document.metadata,
        documents=document.chunks,
    )


def process_one_document(_file: str) -> None:
    try:
        doc = parse_and_clean_document(_file)
        doc = chunk(doc)
        doc = embedding(doc)
        add_document_to_collection(doc)
    except:
        with open(f"{DATA_DIR}/failures_list.txt", "a") as f:
            f.write(f"{_file}\n")
        print(f"{_file} failed")


def process_documents(folder: os.PathLike = FOLDER_DIR, n_processes: int = 1) -> None:
    file_list = os.listdir(folder)
    if n_processes == 1:
        for _file in tqdm(file_list):
            process_one_document(_file)

    else:
        torch.set_num_threads(1)  # set PyTorch mp to 1 thread otherwise it hangs
        with multiprocessing.Pool(processes=n_processes) as pool:
            _ = list(
                tqdm(
                    pool.imap(process_one_document, file_list),
                    total=len(file_list),
                    desc="Processing",
                    unit="file",
                )
            )


if __name__ == "__main__":
    process_documents(n_processes=5)
