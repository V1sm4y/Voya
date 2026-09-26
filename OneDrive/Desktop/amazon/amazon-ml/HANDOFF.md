# HANDOFF: Amazon ML Challenge 2026, Business Entity Resolution

_For teammates who have never seen this project. Last updated after Phase 7 (2026-09-26)._

## 1. The problem (in 5 lines) and the metric
- There are three sources of business records (name, address, country) with no shared ids. **Source 1 (S1)** is deduplicated.
- For **every S1 record**, find **all** records in Sources 2 and 3 that are the same business. There can be 0, 1 or many, including several from the same source.
- Train covers the US and India, with labels. Test adds **France**, with no labels at all.
- Data is noisy: typos, abbreviations, legal-suffix changes, Indic-script transliterations, renamed businesses, reordered or empty addresses, and planted near-duplicate **decoys**.
- Output: one row per test S1 with a comma-separated list of S2/S3 ids.
- **Metric: macro F0.5, averaged per S1.**
  - Precision is weighted twice as heavily as recall, so a wrong merge hurts more than a miss.
  - An S1 with no true match scores 1.0 **only if we predict an empty list**, and 0 otherwise.

## 2. Current status
| item | value |
|---|---|
| **Best CV macro F0.5** (OOF, full train, 2.2M S1) | **0.9741**: US 0.9787, India 0.9672 (flat 0.5 cutoff: 0.9669) |
| Blocking pair recall (full train) | 0.9813 (US 0.9876, India 0.9719), 71 candidates per S1 |
| Pair model OOF PR-AUC | 0.9974 (US ~0.998, India 0.9963) |
| France proxy (model trained on one country, scored on the other) | US→India 0.852, India→US 0.958 |
| Submission files | `output/matching_results.tsv` (96 MB, 1,732,544 rows, 5.67M links) and `output/candidate_pairs.tsv` (1.9 GB, 145.7M scored pairs) |
| Official validator | **PASS** (both files; matching file also with `--check-ids`) |
| submission.zip | `submission.zip` at the repo root (838 MB; CRC OK; gitignored): `output/` (both TSVs) + `code/business_entity_resolution/` + `Documentation_template.md`. **Before uploading, fill in the team name and members at the top of `Documentation_template.md` and rebuild the zip.** |

Phase log (each phase is one commit on `main`):

| phase | result |
|---|---|
| 0 blocking | recall 0.9813 at 71 candidates per S1 (156.8M pairs, 37 min, 8.6 GB) |
| 1 features | 88 → 105 pair features, streamed per partition (~57k pairs/s) |
| 2 model | LightGBM 5-fold; 9.0M training rows from a random 30% of S1s; OOF over every train pair |
| 3 decision | one-S1-per-record assignment + keep/ratio/null thresholds from a stable plateau: 0.9712 |
| 4 error analysis | +name-commonness and sibling (decoy-twin) features: **0.9741**; stopped (gain < 0.003) |
| 5 test submission | 145.7M test pairs scored (71 min); 5.67M links for 1.73M S1; 5.6-5.7% empty in every country; **validator PASS** (also with --check-ids on the matching file) |
| 6 package | methodology filled (`Documentation_template.md`); README with exact commands and timings; `submission.zip` built |

## 3. Repo map and how to run
```
Amazon ML/
├── HANDOFF.md, CLAUDE.md, DATA_NOTES.md, EXPERIMENTS.md, Documentation_template.md (filled methodology)
├── code/business_entity_resolution/   <- ships in the submission zip
│   ├── run_pipeline.py                 entry point: train | predict
│   ├── src/ber/*.py                    pipeline package (table below)
│   ├── tests/                          pytest (107 tests; about 1 s)
│   └── README.md, requirements.txt
│   └── scripts/setup_lightgbm_macos.sh macOS libomp fix for LightGBM
├── reports/                            every evaluation report (markdown)
├── student_resource/                   provided README, validator, template; put the raw data in dataset/
├── data/      (gitignored)             caches, candidates, models, training rows
└── output/    (gitignored *.tsv)       matching_results.tsv, candidate_pairs.tsv
```
**Setup**
```bash
python3 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
PATH=.venv/bin:$PATH bash code/business_entity_resolution/scripts/setup_lightgbm_macos.sh   # macOS only (or: brew install libomp)
# raw data: student_resource/dataset/train/*.tsv and student_resource/dataset/test/*.tsv
```
**Run**
```bash
cd code/business_entity_resolution
../../.venv/bin/python run_pipeline.py train      # ~2.5 h from scratch
../../.venv/bin/python run_pipeline.py predict    # ~1.5 h (blocking + scoring + writing + validator)
../../.venv/bin/python -m pytest tests -q
```

**Caches**
- `data/cache/*.parquet` holds the raw TSVs. It is rebuilt automatically when missing.
- `data/cache/prep/<hash>/` holds the parsed tables. The hash covers `normalize/translit/names/address/prep.py`, so editing any of them triggers an automatic rebuild (about 15 min).
- Candidates (`data/cache/candidates/`) and models (`data/models/`) are **not** hash-keyed. After changing blocking, pass `--force-blocking`; after changing features, rerun `ber.train` (collect → fit → oof) and then `ber.decide`.
- **Splits are frozen.** Don't rebuild `geo`/`splits` casually, because it moves DEV.

| module (`src/ber/`) | what it does |
|---|---|
| `data`, `paths`, `cli` | parquet cache, sample masks, paths and env overrides, the `--sample dev\|full` flag |
| `normalize`, `translit`, `names`, `address` | normalisation; Indic→Latin romaniser + phonetic key; core name + legal form + alias; address parser (India-specific) |
| `prep` | parsed per-record table cached by code hash |
| `geo`, `splits` | state/city per record; 5 folds grouped by S1; geographic DEV; cross-country splits |
| `blocking`, `blocking_tune` | 13 blockers → candidates (global int indices); offline K tuning |
| `features` | 105 pair features streamed per (country, state group) partition |
| `train` | training-row sampling → LightGBM fold models → OOF for every pair |
| `decide` | assignment + thresholds; exact vectorised macro F0.5; stable grid choice |
| `errors` | OOF error categories with oracle gains |
| `predict` | test scoring → streaming writer → validator |
| `metrics`, `submission` | the exact metric; format-safe writers |

## 4. Key findings that shape the design (details in DATA_NOTES.md)
- **One S1 per record.** Every S2/S3 record belongs to at most one S1. The decision layer gives each record only to its highest-probability S1, which is worth a lot on singletons.
- **Decoys.** About 26% of S2/S3 records match nothing, and many are planted near-duplicates: an extra word, a swapped legal form, or a house number ±1–15. Legal form is kept as a feature (never dropped, never a hard rule). House numbers use *soft* features, because true matches are also truncated or padded. Sibling features compare a candidate with the S1's other candidates, where the real twin usually is.
- **Indic scripts.** 13–24% of India matches have Indic-script names. `translit.py` romanises nine scripts, and a phonetic skeleton gives one key across scripts and spellings. State names in Indic scripts are often translations.
- **Address-only matches.** 5–8% of matches have an unrelated name (brand, domain, handle, initials). Address-only blockers and features catch them.
- **Test shift.**
  - Test has more candidates per S1: France 127, India 95, US 54, against train's 94 / 56. Context features are relative (ranks, gaps, z-scores), so they survive the shift.
  - France is unlabelled, and its S2/S3 often drop the region. The cross-country proxy (0.85–0.96) suggests France will score below CV.
- **DEV is biased.** The geographic DEV (whole cities) is easy for India, because its one drawn metro (Jaipur) is easy. **Use DEV only for A/B deltas.** Absolute numbers and thresholds come from full-train OOF.

## 5. What's left (prioritised)
Expected gains are OOF oracle upper bounds or informed guesses. "∥" means the task can run in parallel with the other ∥ tasks without touching the same files.

| # | task | owner | effort | expected gain | parallel? / files |
|---|---|---|---|---|---|
| 1 | **France-specific normalisation**: department↔region inference from the city, French street abbreviations (R./AV/BD/CHE), SARL/SAS/EURL legal forms, accents; eyeball 200 test France predictions; tighten `place_premise` for France (French names are generic, 32 candidates per S1 from that blocker alone) | unassigned | 1–2 d | large on the 15% France share of test (unmeasurable offline) | ∥ `address.py`, `names.py`, `blocking.py` (France branch) |
| 2 | **Threshold tuning against the leaderboard**: current keep 0.7 / ratio 0.8 / null 0.7 come from US+India OOF; try null 0.6 / 0.8 on the public LB, especially for France | unassigned | 0.5 d | 0.002–0.01 | ∥ `data/models/decision.json` only |
| 3 | **Empty-address and low-p misses** (oracle 0.0057 each): name-only evidence; add target-side commonness of the *full* name + S1-side uniqueness; per-source calibration | unassigned | 1 d | ≤ 0.005 | ∥ `features.py` (FREQ_F), then retrain |
| 4 | **Blocking misses** (oracle 0.0070): India residual (Indic names + noisy addresses, typo'd generic names); consider a phonetic TF-IDF with larger K for India only | unassigned | 1–2 d | ≤ 0.004 | `blocking.py` (not ∥ with 1) |
| 5 | **India metro errors** (Delhi districts, Howrah, Patna are the weakest): locality/landmark alignment, district↔city aliasing | unassigned | 1 d | ~0.002 | ∥ `address.py` (India branch), coordinate with 1 |
| 6 | **Speed/memory**: cache test feature partitions; parallelise partitions over processes (8.6 GB peak now); a full retrain + OOF takes about 1.5 h | unassigned | 1 d | faster iteration | ∥ `features.py`, `train.py` |
| 7 | Extra features: landmark text similarity, floor equality, S2 vs S3 source-specific noise flags, second-best-sibling gaps | unassigned | 0.5–1 d | ~0.001–0.002 | `features.py` (not ∥ with 3) |
| 8 | Second Phase-4 iteration on the remaining categories (decoys 0.0044) | unassigned | 0.5 d + 1.5 h compute | ~0.002 | `features.py` |

## 6. Rules and pitfalls
- **No external data, APIs, geocoding or registry lookups.** Hand-written general knowledge (state names, abbreviations) is fine. The model must be MIT/Apache and ≤ 8B parameters (LightGBM is MIT).
- **Format:**
  - The header is exactly `source1_entity_id<TAB>matched_entity_ids`, with **one row per test S1** (empty = no match).
  - Ids are joined with `,` and no spaces; files are UTF-8 with `\n` line endings; only S2-/S3- ids; no duplicates.
  - `candidate_pairs.tsv` must contain every matched id.
  - Always write through `ber.submission` and run `student_resource/utils/validate_submission.py`.
- **Never commit `data/`, raw TSVs, `*.parquet`, `*.zip` or `output/*.tsv`** (see `.gitignore`), and nothing larger than 50 MB. The missed-pair dumps in `reports/*.tsv` contain raw records and are ignored too.
- Never use raw country as a model feature (France must transfer). No hard-rule rejections; everything is a feature.

## 7. Experiments
See **EXPERIMENTS.md** for every change with before/after scores. The reports behind each number are in `reports/`: `blocking_*`, `model_report`, `feature_importance`, `decision_report`, `error_analysis`, `predict_test_full`.

## 8. How to rebuild submission.zip (after editing the doc or re-predicting)
```bash
cd "Amazon ML"   # repo root
.venv/bin/python - <<'PY'
import zipfile; from pathlib import Path
files = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"]
files += [str(p) for p in sorted(Path("code").rglob("*")) if p.is_file() and "__pycache__" not in p.parts
          and ".pytest_cache" not in p.parts]
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
    for f in files: z.write(f, f)
PY
```
