#!/usr/bin/env python
"""End-to-end entry point (no network, fixed seeds).

  python run_pipeline.py train     # cache -> prep -> geo/splits -> blocking(train) -> features/model -> decision
  python run_pipeline.py predict   # blocking(test) -> features -> model -> decision -> output/*.tsv -> validator

Each stage skips work whose cached output already exists (data/cache, data/models). Add --force-blocking to
recompute candidates. Paths: see src/ber/paths.py (override with BER_DATA_DIR / BER_WORK_DIR / BER_OUTPUT_DIR).
Run from this folder with the project venv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def run(*args):
    print("$ python -m", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", *args], cwd=SRC)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(args)} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--force-blocking", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(SRC))
    from ber import data, paths, prep
    from ber.blocking import candidates_path
    data.build_cache()
    prep.build()
    if a.mode == "train":
        if not (paths.SPLITS_DIR / "train_folds.parquet").exists():
            run("ber.geo")
            run("ber.splits")
        if a.force_blocking or not candidates_path("train_full").exists():
            run("ber.blocking", "--sample", "full")
        run("ber.train", "--sample", "full")
        run("ber.decide", "--sample", "full")
    else:
        if a.force_blocking or not candidates_path("test_full").exists():
            run("ber.blocking", "--sample", "full", "--split", "test")
        run("ber.predict", "--sample", "full", "--split", "test")


if __name__ == "__main__":
    main()
