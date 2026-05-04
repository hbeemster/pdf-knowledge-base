"""Optimize RAG with Optuna."""

import os
import time
from multiprocessing import Pool

import optuna
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from optuna import Trial
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from pdf_knowledge_base.constants import (
    PDF_EXAMPLE,
    EMBEDDINGS_FOLDER,
    CHROMA_FOLDER,
    OPTUNA_FOLDER,
    StorageType,
)
from pdf_knowledge_base.ingest import pdf_to_documents, split_documents

optuna.logging.enable_propagation()

logger.info("Starting RAG with Optuna.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder=str(EMBEDDINGS_FOLDER / "optuna"),
)

queries = [
    "In what way can anthropologists bring value to software engineering and computer science?",
    "What are the risks of autonomous agents that have access to tools?",
]


def objective(trial: Trial):
    """RAG with Optuna."""
    logger.info(f"Start trial: {1 + trial.number}")
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
    chroma_dir = f"{trial.study.user_attrs['chroma_path']}/{param_desc}"
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
        total_score += score

    # average score
    avg_score = total_score / len(queries)
    return avg_score


# ------------------------------------------------------------------------
def optimize(study_name: str, storage: JournalStorage, n_trials: int = 3) -> None:
    """Optimize RAG with Optuna."""
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )
    tmpdir = str(CHROMA_FOLDER / f"{study_name}")
    study.set_user_attr("chroma_path", tmpdir)

    study.optimize(objective, n_trials=n_trials)


# ------------------------------------------------------------------------
def run_optimizer(
    study_name,
    n_processes: int = None,
    n_trials: int = 3,
):
    if n_processes is None:
        n_processes = os.cpu_count() or 1
    n_processes = min(n_processes, n_trials)

    base, remainder = divmod(n_trials, n_processes)
    trials_list = [base + 1 if n < remainder else base for n in range(n_processes)]

    file_path = f"{OPTUNA_FOLDER}/journal.log"
    lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)
    storage = JournalStorage(
        JournalFileBackend(file_path=file_path, lock_obj=lock_obj)
    )

    start_time = time.time()
    with Pool(processes=n_processes) as pool:
        pool.starmap(
            optimize, [(study_name, storage, trials) for trials in trials_list]
        )
    elapsed = time.time() - start_time

    logger.info(75 * "=")
    study = optuna.load_study(study_name=study_name, storage=storage)
    logger.info(
        f"Best result with value {study.best_trial.value} and params {study.best_params}"
    )
    logger.info(f"Total run time: {elapsed:.2f}s")
    logger.info(75 * "=")


# ------------------------------------------------------------------------
def cli_optimize():
    """CLI entry point for run_optimizer."""
    import typer
    from typing import Optional

    def _main(
        study_name: str,
        n_trials: int = typer.Option(3, help="Number of trials"),
        n_processes: Optional[int] = typer.Option(None, help="Parallel processes (default: cpu count)"),
    ):
        run_optimizer(study_name=study_name, n_trials=n_trials, n_processes=n_processes)

    typer.run(_main)


# ------------------------------------------------------------------------
if __name__ == "__main__":
    # optimize()
    run_optimizer(study_name="optimize-rag", n_trials=2)
