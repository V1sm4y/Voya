"""Filesystem locations. Override with env vars when the repo layout differs (e.g. the submission zip).

BER_DATA_DIR   folder holding train/ and test/ TSVs   (default: <root>/student_resource/dataset)
BER_WORK_DIR   folder for caches, splits, dummy files (default: <root>/data)
BER_OUTPUT_DIR folder for submission files            (default: <root>/output)
"""
import os
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "student_resource").is_dir() or (p / "CLAUDE.md").is_file():
            return p
    return Path.cwd()


ROOT = _find_root()
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "data"))
CACHE_DIR = WORK_DIR / "cache"      # parquet copies of the raw TSVs (gitignored)
SPLITS_DIR = WORK_DIR / "splits"    # fold / dev-sample id lists (small, deterministic)
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))
REPORTS_DIR = ROOT / "reports"      # evaluation reports (markdown)
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
