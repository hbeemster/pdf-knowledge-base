import json
from pathlib import Path
from typing import List

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from loguru import logger

from pdf_knowledge_base.constants import (
    PDF_EXAMPLE,
    JSON_FOLDER,
    EMBEDDINGS_FOLDER,
    CHROMA_FOLDER,
)


# ------------------------------------------------------------------------
# public functions
# ------------------------------------------------------------------------
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True,
) -> List[Document]:
    """"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    return text_splitter.split_documents(docs)


# ------------------------------------------------------------------------
def ingest_pdfs(pdf_file: Path = PDF_EXAMPLE):

    documents = pdf_to_documents(pdf_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    docs = split_documents_2(documents, text_splitter)
    embeddings = get_embeddings()
    embeddings.embed_documents([doc.page_content for doc in docs])

    vector_store = store_embeddings(documents, embeddings)


# ------------------------------------------------------------------------
def pdf_to_documents(pdf_file: Path, json_folder: Path = JSON_FOLDER) -> List[Document]:
    if not pdf_file.exists():
        raise ValueError(f"File {pdf_file} does not exist.")
    json_filename = json_folder / f"{pdf_file.stem}.json"
    if json_filename.exists():
        logger.debug("Read from JSON")
        with open(json_filename) as f:
            docs = [Document(**d) for d in json.load(f)]
    else:
        from langchain_community.document_loaders import PyPDFLoader

        logger.debug("Load PDF")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        logger.debug("Done")
        logger.debug("Persist as JSON")

        with open(json_filename, "w") as f:
            json.dump(
                [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in docs
                ],
                f,
            )

    logger.debug("Done")
    return docs


# ------------------------------------------------------------------------
def split_documents_2(
    docs: List[Document],
    text_splitter: TextSplitter = None,
) -> List[Document]:
    logger.debug("Split documents into chunks")
    docs_ = text_splitter.split_documents(docs)
    logger.debug("Done")
    return docs_


# ------------------------------------------------------------------------
def get_embeddings(embeddings_folder: Path = EMBEDDINGS_FOLDER) -> Embeddings:
    logger.debug("Embed documents")

    store = LocalFileStore(embeddings_folder)

    underlying_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(embeddings_folder),
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model_name,
    )

    logger.debug("Done")
    return cached_embeddings


# ------------------------------------------------------------------------
def store_embeddings(docs: List[Document], embeddings: Embeddings) -> VectorStore:

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_FOLDER),
    )
    ids = vector_store.add_documents(documents=docs)

    return vector_store


# ------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_pdfs()
