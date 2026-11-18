"""Paths for project code and the existing LatentSync 1.5 engine."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BACKEND = Path(os.environ.get('MUSICTALK_LATENTSYNC_DIR', str(ROOT))).resolve()
WORKER_PYTHON = BACKEND / '.venv' / 'Scripts' / 'python.exe'
