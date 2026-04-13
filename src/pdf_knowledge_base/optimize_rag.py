"""Optimize RAG with Optuna."""
from datetime import datetime

import optuna
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from optuna import Trial

from pdf_knowledge_base.constants import PDF_EXAMPLE, EMBEDDINGS_FOLDER, CHROMA_FOLDER, OPTUNA_FOLDER
from pdf_knowledge_base.ingest import pdf_to_documents, split_documents

optuna.logging.enable_propagation()

logger.info("Starting RAG with Optuna.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder=str(EMBEDDINGS_FOLDER / "optuna"),
)
# vector_store = InMemoryVectorStore(embeddings)

# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory=str(CHROMA_FOLDER / "optuna"),
# )
#

queries = [
    "In what way can anthropologists bring value to software engineering and computer science?",
    "What are the risks of autonomous agents that have access to tools?",
]


def objective(trial: Trial):
    """RAG with Optuna."""
    logger.info(f"Start trial: {trial.number}")
    documents = pdf_to_documents(PDF_EXAMPLE)

    # Set up trial parameters
    chunk_size = trial.suggest_int("chunk_size", 800, 1200)
    logger.info(f"{chunk_size=}")
    chunk_overlap = trial.suggest_int("chunk_overlap", 170, 230)
    logger.info(f"{chunk_overlap=}")

    # Execute trial
    splits = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    param_desc = f"{trial.number}-{chunk_size}-{chunk_overlap}"
    chroma_dir = f"{trial.study.user_attrs["tmp_chroma_path"]}/{param_desc}"
    vector_store = Chroma(
        collection_name=param_desc,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    vector_store.add_documents(documents=splits)

    # Evaluate trial
    total_score = 0
    for query in queries:
        docs_and_scores = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=3,
        )

        doc, score = docs_and_scores[0]
        logger.info(
            f"Source: {doc.metadata['source']}, page_label: {doc.metadata['page_label']}"
        )
        logger.info(f"Score: {score}")
        total_score += score

    # average score
    avg_score = total_score / len(queries)
    logger.info(f"Avg score: {avg_score}")
    return avg_score


# ------------------------------------------------------------------------
def optimize():
    """Optimize RAG with Optuna."""
    study_name = "optimize-rag-3"  # Unique identifier of the study.
    database_name =f"{OPTUNA_FOLDER}/{study_name}.db"
    storage = f"sqlite:///{database_name}"
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )
    tmpdir = str(CHROMA_FOLDER / f"{study_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    study.set_user_attr("tmp_chroma_path", tmpdir)

    study.optimize(objective, n_trials=15)
    logger.info(study.best_params)


if __name__ == "__main__":
    optimize()
