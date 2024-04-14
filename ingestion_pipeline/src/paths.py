import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent.parent
DATA_DIR = PARENT_DIR / "inference_pipeline/data"
FOLDER_DIR = DATA_DIR / "lettres_de_suivi"

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)
