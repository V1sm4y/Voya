# business_entity_resolution

End-to-end pipeline for the Amazon ML Challenge 2026 **Business Entity Resolution** task. For every Source-1
record it finds all matching Source-2/3 records, scored by macro F0.5 per S1. No external data or services are
used. See `../../HANDOFF.md` for results and `../../Documentation_template.md` for the methodology.

## 1. Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# macOS without Homebrew: LightGBM needs libomp; point it at scikit-learn's bundled copy (not needed on Linux)
PATH=.venv/bin:$PATH bash scripts/setup_lightgbm_macos.sh   # or: brew install libomp
```
Put the competition data at `student_resource/dataset/{train,test}/*.tsv`, relative to the repo root, or set
`BER_DATA_DIR`. `BER_WORK_DIR` (caches, models) defaults to `<repo>/data`, and `BER_OUTPUT_DIR` defaults to
`<repo>/output`.

## 2. Run end to end
```bash
cd code/business_entity_resolution
../../.venv/bin/python run_pipeline.py train     # cache -> prep -> geo/splits -> blocking -> features/model -> decision
../../.venv/bin/python run_pipeline.py predict   # test blocking -> features -> model -> decision -> output/*.tsv -> validator
```
Each stage reuses cached outputs (`data/cache`, `data/models`). Approximate timings on a 10-core / 16 GB laptop:

| stage | module | time |
|---|---|---|
| TSV→parquet cache | `ber.data` | 1 min |
| prep (normalise, parse, transliterate) | `ber.prep` | 15 min (both splits) |
| blocking, train | `ber.blocking --sample full` | 37 min |
| training rows | `ber.train --stage collect` | 17 min |
| fit (5 folds + 2 cross-country models) | `ber.train --stage fit` | 10 min |
| OOF over every train pair | `ber.train --stage oof` | 51 min |
| decision grid | `ber.decide` | 5 min |
| blocking, test | `ber.blocking --sample full --split test` | ~40 min |
| predict + write + validate | `ber.predict` | ~50 min |

Low-disk design: pair features are never stored for all pairs. They are streamed per (country, state group)
partition into the consumer (training-row sampler, OOF scorer or test scorer). Only the training rows (~0.5 GB),
the OOF pairs with p ≥ 0.02 (~0.2 GB) and the candidate lists (~0.9 GB zstd) are written.

## 3. Individual stages (from `src/`)
```bash
python -m ber.splits                                  # folds + geographic DEV (deterministic)
python -m ber.prep                                    # prep cache, keyed by normalisation-code hash
python -m ber.blocking --sample dev|full [--split test]
python -m ber.train --sample full --stage collect|fit|oof
python -m ber.decide --sample full                    # threshold grid on OOF -> data/models/decision.json
python -m ber.errors --sample full                    # OOF error analysis -> reports/error_analysis.md
python -m ber.predict --sample full --split test      # submission files + validator
python -m pytest ../tests -q
```

## 4. Modules (`src/ber/`)
| module | role |
|---|---|
| `paths.py`, `cli.py` | locations / env overrides; shared `--sample dev|full --split` args |
| `data.py` | TSV→parquet cache, sample masks, truth |
| `geo.py`, `splits.py` | per-record state/city; 5 folds grouped by S1, cross-country splits, geographic DEV |
| `normalize.py`, `translit.py`, `names.py`, `address.py` | normalisation, Indic romanisation and phonetic key, name (core + legal form + alias) and address parsing |
| `prep.py` | per-record parsed table, cached by code hash |
| `blocking.py`, `blocking_tune.py` | 13 blockers per country and the recall report; offline K tuning |
| `features.py` | 105 pair features, streamed per partition |
| `train.py` | training-row sampling, LightGBM fold models, OOF scoring |
| `decide.py` | one-S1-per-record assignment + thresholds, exact fast macro F0.5 |
| `errors.py` | OOF error categories with oracle gains |
| `predict.py` | test scoring, streaming submission writer, validator |
| `metrics.py`, `submission.py` | exact metric; format-safe writers |
