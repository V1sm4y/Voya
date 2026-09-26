# Amazon ML Challenge 2026 — Business Entity Resolution

## Task
For every **Source 1** record (the deduplicated reference, `S1-*`), find **all** records in **Source 2** (`S2-*`) and **Source 3** (`S3-*`) that refer to the same real-world business. There can be 0, 1 or many such records, and several can come from the same source. The sources share no ids. Each record has the fields `entity_id, business_name, business_address, country`.
- Train covers US and India, with labels in `train_ground_truth.tsv` (one row per S1 → comma-separated S2/S3 ids; an empty value means no match).
- Test adds **France** (unlabelled). Treat `country` as an open set: never hard-code `{US, India}`.
- Every S2/S3 record belongs to **at most one** S1 (the match sets are disjoint), and about 26% of S2/S3 records belong to none (decoys and distractors). See `DATA_NOTES.md` for all EDA findings and read it before designing anything.

## Metric
The score is **macro F0.5 per S1 entity**, averaged over all test S1 records:
`F0.5 = 1.25·P·R / (0.25·P + R)`.
- An S1 with no true match scores 1.0 only if we predict an **empty** list, and 0 otherwise.
- The metric is precision-heavy: a false merge costs more than a missed link.
- Tune thresholds on a held-out validation split made of whole S1 clusters plus a proportional share of unlinked S2/S3 records.

## Hard rules
- **No external data.** No external databases, APIs, geocoding, business-registry lookups or internet data augmentation; breaking this means disqualification. Only the provided files may be used. Offline, rule-based code such as transliteration libraries and hand-written abbreviation maps is fine.
- The final model must be MIT or Apache-2.0 licensed and have ≤8B parameters.
- All inputs and outputs are **TSV** (`sep="\t"`). Always read with `dtype=str, keep_default_na=False` so that empty strings are not turned into NaN.

## Outputs (see DATA_NOTES.md §5)
- `output/matching_results.tsv` has the header `source1_entity_id<TAB>matched_entity_ids`. It needs one row for every test S1, with ids comma-separated with no spaces, left empty when there is no match, and only S2/S3 ids.
- `output/candidate_pairs.tsv` has the header `source1_entity_id<TAB>candidate_entity_ids`. It holds the final candidate set the model scores, and every matched id must be in it.
- Before submitting, validate from `student_resource/`:
  `python3 utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test`
- The final zip contains `output/`, `code/business_entity_resolution/{src/,README.md,requirements.txt}` and the filled-in `Documentation_template.md`.

## Project layout
```
Amazon ML/
├── CLAUDE.md                      # this file
├── DATA_NOTES.md                  # EDA findings (schemas, GT structure, noise patterns, format rules)
├── student_resource/              # provided data + docs — READ-ONLY, never modify
│   ├── dataset/{train,test}/*.tsv
│   ├── utils/validate_submission.py
│   ├── Documentation_template.md
│   └── README.md
├── code/
│   └── business_entity_resolution/  # the pipeline (this folder ships in the submission zip)
│       ├── src/ber/                 # package: paths, data, geo, splits, metrics, submission, normalize,
│       │                            #   translit, names, address, baselines, parse_report, cli
│       ├── tests/                   # pytest: metric, writer + official validator, translit, names, address, splits
│       ├── README.md                # exact reproduce steps
│       └── requirements.txt         # pinned deps
├── data/                          # generated working data
│   ├── cache/                     # parquet copies of every TSV + geo_<split>_source<n>.parquet (gitignored)
│   ├── splits/                    # train_folds.parquet, dev_cities_{train,test}.parquet, test_dev.parquet
│   └── dummy/                     # dummy submission used to prove the validator passes (gitignored)
├── reports/                       # evaluation reports written by scripts (e.g. baselines_dev.md)
├── output/                        # matching_results.tsv, candidate_pairs.tsv (generated, not hand-edited)
├── eda/                           # one-off EDA scripts behind DATA_NOTES.md (read data/cache/ by default)
└── notebooks/                     # optional exploration
```

## Data cache
- `data/cache/<name>.parquet` holds the raw TSVs converted once, with row counts checked against the line counts. It is **gitignored**, and it is built automatically by `ber.data.load()` if missing, or explicitly with `.venv/bin/python eda/load.py [--force]`.
- `data/cache/geo_<split>_source<n>.parquet` holds the per-record (state, city) that `ber.geo` computes from the address parser in about 70 s on 10 cores. When a record has no state, the state is inferred from the S1 city → state map of the same split. DEV sampling depends on this cache. **Rebuild it with `python -m ber.geo --force` whenever `address.py`'s state or city logic changes, then rebuild the splits.**
- **Prep cache** (`ber.prep`): `data/cache/prep/<code-hash>/<split>_source<n>.parquet` holds one row per record with the name fields, the address fields and the blocking strings (name_tx, places, addr_tx...), in raw-cache row order.
  - The hash covers `normalize.py, translit.py, names.py, address.py, prep.py`. Editing any of them rebuilds the tables on next use (about 15 min for both splits); nothing else does.
  - `python -m ber.prep --prune` deletes the tables of old hashes.
  - Read them with `prep_table(split, n, columns=[...])` (pyarrow-backed strings), and select a sample with the row masks from `ber.data.sample_masks(split, sample)`.
- **Splits are frozen.** DEV membership comes from the geo cache, which is *not* hash-keyed. Don't rebuild `geo` or `splits` just because `address.py` changed, or DEV moves and A/B comparisons break.
- `data/cache/candidates/<split>_<sample>.parquet` holds the blocking output (s1, target, blocker bits). The `raw_*` and `truth_*` files come from `--sweep` runs, for `ber.blocking_tune`.
- Always load data through `ber.data.load(split, sample)` (raw records) or `prep_table` + `sample_masks` (parsed), never with ad-hoc `read_csv`.

## Working conventions
- Run pipeline modules from `code/business_entity_resolution/src` as `python -m ber.<module>`, and run the tests with `python -m pytest ../tests -q`.
- **Every script must accept `--sample dev|full`** (use `ber.cli.base_parser`), and iterate on `dev` first.
  - **DEV is geographic** (`ber.splits.sample_cities`, seed 2027).
    - Cities are grouped by (country, canonical city) and tiered by size within each country: metro ≥1% of the country's S1s, mid ≥0.05%, small is the rest.
    - In each tier, random cities are drawn until about 5% of the tier's S1s are covered, with at least one city per tier.
    - **DEV S1** is every S1 in the drawn cities: 113,463 S1s (5.14%), namely India 4.0% (metro: Jaipur) and US 5.9% (metro: Houston). They are spread evenly over the folds, and the singleton share is 0.0559, the same as all of train.
    - **DEV pool** is every S2/S3 record in the same (country, state group), plus the records with an unknown state. Records with a wrong or missing city are therefore kept, and 100% of DEV true links are in the pool. Andhra Pradesh and Telangana form one state group (Hyderabad).
    - Whole cities are in DEV, so the S1s that compete for the same records are present too. **No extra-S1 workaround is needed** for competition-aware steps such as assigning each S2/S3 record to its single best S1.
    - Cost: the sampled towns span nearly every state, so the DEV pool is about 10.3M records, nearly the full pool. DEV saves time on the S1 side only.
  - Test DEV uses the same geographic draw on test: 85,000 S1s, including 7,833 in France. It is meant for smoke tests.
  - The old 5% random DEV is kept only as `sample="dev_random"` inside `ber.data.load` (and `ber.baselines --legacy-random-dev`) for comparisons. Don't build new work on it.
- **Splits** (`ber.splits`, seed 2026) are grouped by S1 id:
  - `kfold()` gives 5 folds stratified by country × size.
  - `cross_country()` gives US→India and India→US, as a proxy for transfer to unlabelled France.
  - Never split the S2/S3 pool. Match against all S2/S3 records of the country and score only the held-out S1 ids.
- **Names: never drop the legal form.** Use `ber.names.parse_name`. It returns `core_name` (legal form removed; exact blocking keys may use it), `core_key` (cross-script phonetic key), `legal_form` (canonical, '+'-joined, including forms written in Indic scripts) and `alias` (the f/k/a or t/a part). Features must compare forms with `names.legal_relation` (same / subset / overlap / conflict / missing), because the US decoys differ only in LLC vs Corp.
- **Addresses: use `ber.address.parse_address(addr, country)`.** It gives one schema for every country: state, city, locality, street, building, premise_no / premise_typed, unit, floor, landmark, admin, careof and pin, plus phonetic `locality_key` / `street_key` / `building_key`.
  - It does the India-specific work: sector/phase/block, nagar/colony/layout, landmark phrases, typed building/plot/flat/shop/khasra/survey numbers, and c/o.
  - Indic components are romanised with `ber.translit`. Its phonetic key gives the same key across scripts and spellings (Kancheepuram = Kanchipuram, महाराष्ट्र = Maharashtra). It is a similarity key, not proof of identity.
- Geography lists (states, French departments, old/new city names) are hand-written general knowledge, which the no-external-data rule allows. Don't replace them with anything fetched.
- **Score only with `ber.metrics.report`**, the exact macro F0.5. It breaks the score down into overall, singleton vs non-singleton, by country and by true size. Write reports to `reports/`.
- **Write submissions only with `ber.submission.write_submission`.** It enforces the header, one row per S1, `,` joins with no spaces, `\n` line endings, UTF-8, and matches ⊆ candidates. Then run the official validator (`ber.submission.run_validator` or `ber.make_dummy`).

## Current baselines (train, macro F0.5)
Geographic DEV (`python -m ber.baselines --sample dev`). `w` is the stratum-weighted score, which is the headline DEV number:

| baseline | overall | overall (w) | singleton | non-singleton | India | US |
|---|---|---|---|---|---|---|
| all-empty | 0.0559 | 0.0557 | 1.000 | 0.000 | 0.0556 | 0.0560 |
| exact core_name + same city | 0.5762 | 0.5701 | 0.665 | 0.571 | 0.523 | 0.601 |
| … + no legal-form conflict | 0.5984 | 0.5902 | 0.800 | 0.587 | 0.533 | 0.628 |

The same code on the old random DEV (unbiased for all of train) gives 0.0559 / 0.5432 / 0.5630, with India at 0.468 / 0.477 and US at 0.593 / 0.621.

**How to read DEV numbers:**
- **Use DEV for A/B deltas, not absolute levels.** The legal-conflict filter gains +0.020 on the random DEV and +0.020 (weighted) on the geographic DEV, so the deltas agree.
- **The absolute level is optimistic for India, by about +0.03 overall.**
  - India metros hold 55% of India's S1s, and one draw has to stand in for all 21 of them.
  - The draw was Jaipur, one of the easiest (0.53, against Mumbai 0.39 and the Delhi districts 0.35–0.39; the std across metros is 0.06).
  - The stratum weights fix the country and tier mix but cannot fix which metro was drawn.
- **Take thresholds and headline numbers from full-data 5-fold CV.**
- Fold-to-fold std on DEV is about 0.002–0.003, so treat A/B differences below about 0.005 as noise.

## Pipeline status
- **Everything past blocking is done:** features (105), LightGBM (5-fold), decision layer, test submission.
- **CV macro F0.5 is 0.9741** (US 0.9787, India 0.9672); the validator passes.
- See **HANDOFF.md** for the team-facing summary, commands and task list, and **EXPERIMENTS.md** for the score log.
- The entry point is `code/business_entity_resolution/run_pipeline.py train|predict`.

## Blocking (`ber.blocking`)
- **Blockers:** 13 in total, always inside one country, unioned. See the module docstring.
  - Ranked (top-K, rank kept): name TF-IDF, phonetic-skeleton TF-IDF, address TF-IDF, name TF-IDF over placeless targets.
  - Exact keys: rare token + place, phonetic name + place, house number + street + place, (place, premise) with a loose-name filter, the India locality keys, placeless rare token, state-level safety-net keys, and S3 aliases.
- **Every key that contains a number has a number-free backup.**
- **Defaults** were tuned on DEV with `blocking_tune`, at the knee of the Pareto front: K = name 10, addr 10, phon 3, placeless 2, alias 1.
  - Generation takes about 14 min on DEV. Nearly all of it is target-side work, because the DEV pool is about the full pool.
- **DEV results:**
  - Pair recall 0.9819 (India 0.9729, US 0.9861; S2 and S3 alike) at 53.8 candidates per S1 (India 67, US 48; p99 217).
  - 91.8% of India and 95.4% of US non-singleton S1s have their full match set covered.
  - India metros: 0.959 (Patna) to 0.987 (Kolkata); Mumbai 0.964.
- **Full train (2.2M S1, 42 min, peak RSS 8.4 GB):**
  - Pair recall **0.9798** (India 0.9703, US 0.9861) at 62 candidates per S1 (India 85, US 47), 137M pairs.
  - Full match sets are covered for 91.2% of India and 95.4% of US S1s.
  - India metros: 0.959 (Patna) to 0.982 (Ernakulam); Mumbai 0.965.
  - DEV is about 0.002 optimistic, as expected for India.
- **Test DEV (candidates only):** US 45, India 86 (p90 182), **France 117** candidates per S1. France's `place_premise` is loose because French names are generic words; there are no labels to tune it. A full test run hasn't completed yet.
- **Next config to verify:** `k_name=15` (DEV 0.9836 at 65 candidates per S1, full estimated ~0.981). It is not yet the default.
- **Disk constraint:** full runs need about 10 GB of free disk for swap. At 98% disk full, macOS can't grow swap, and the runs stall in I/O wait (a 14 s top-K call took 900 s).
  - `_combos` now hashes per column instead of building key strings, which removes the main memory spike (full train had reached a 26 GB footprint). This is covered by unit tests but hasn't yet been re-run on full data.
- **The India ≥ 0.98 target is not reachable by raising K** (0.9798 at 142 candidates per S1). The residual misses are Indian-script names with noisy addresses, typo'd names at number-only addresses, and generic names ranked out.
- **Most valuable blockers (unique catches):** address TF-IDF (4.0% of true pairs), name TF-IDF (0.74%), and `place_premise` (0.67%, and −1.8% India recall if dropped).
- Reports: `reports/blocking_*.md`, `reports/blocking_tune_train_dev.md`, and the missed pairs with a likely reason in `reports/blocking_missed_*.tsv`.

## Environment
- The system `python3` (3.9) has a broken pandas/numpy install. **Use the project venv at `.venv/bin/python`**. It has pandas 2.2.3, numpy 2.0.2, pyarrow, rapidfuzz, tabulate and pytest, and the pins are in `code/business_entity_resolution/requirements.txt`. Add new dependencies to the venv and pin them in `requirements.txt`.
- The machine has 16 GB RAM and 10 cores. The full data is about 2.4 GB of TSV (about 23M rows). Convert to parquet once (reading takes about 3–8 s per file) and avoid Python-level loops over 10M rows. Use vectorised or rapidfuzz `cdist` code, and process work per country or per block.
