"""Constants."""
from pathlib import Path


# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[2]
DATA_FOLDER = PROJECT_ROOT / "data"
PDF_FOLDER = DATA_FOLDER / "pdf"
PDF_EXAMPLE = PDF_FOLDER / "cacm-2026-01.pdf"
JSON_FOLDER = DATA_FOLDER / "json"
EMBEDDINGS_FOLDER = DATA_FOLDER / "embeddings"
CHROMA_FOLDER = DATA_FOLDER / "chroma"
OPTUNA_FOLDER = DATA_FOLDER / "optuna"


PDF_FOLDER.mkdir(parents=True, exist_ok=True)
JSON_FOLDER.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)
