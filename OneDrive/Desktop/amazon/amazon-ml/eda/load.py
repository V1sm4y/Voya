"""Build the parquet cache (data/cache/<name>.parquet) from student_resource/dataset/*.tsv.

Thin wrapper around ber.data.build_cache so EDA and the pipeline share one cache.
Usage: .venv/bin/python eda/load.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
from ber import data, paths  # noqa: E402

data.build_cache(force="--force" in sys.argv)
print("cache:", paths.CACHE_DIR)
