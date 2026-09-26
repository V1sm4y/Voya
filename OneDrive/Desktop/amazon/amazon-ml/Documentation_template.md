# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ⟨fill in team name⟩
**Team Members:** ⟨fill in team members⟩
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
We built a four-stage pipeline:

1. **Normalisation**, including rule-based transliteration of nine Indic scripts.
2. **Recall-oriented blocking**, with 13 blockers inside each country.
3. **A LightGBM pair classifier** on 105 string, address, soft-number and context features.
4. **A decision layer** that gives every Source-2/3 record to at most one S1 and applies thresholds tuned on out-of-fold predictions with the exact macro F0.5.

Out-of-fold CV on the full training set gives **macro F0.5 = 0.9741** (US 0.9787, India 0.9672). Blocking pair recall is 0.981. No external data, APIs or geocoding are used.

---

## 2. Methodology

### 2.1 Problem Analysis
From EDA of 2.2M train S1 records, 10.3M S2/S3 records and 7.64M labelled links:

- **The mapping is a partition.** Every S2/S3 record belongs to at most one S1. 5.6% of S1s have no match, and 77% have several matches from the same source (up to 11 in total).
- **Decoys.** About 26% of S2/S3 records match no S1, and many are planted near-duplicates:
  - the same name plus one extra word ("+ Industries", "[Holdings]");
  - a different legal form (LLC vs Corp, Private vs Public Ltd);
  - the same street with a house number shifted by 1–15.
- **Scripts and renames.**
  - 13–24% of India matches write the name in an Indic script (Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia).
  - 5–8% of matches carry an unrelated name (a brand, domain, handle or initials) and can only be matched on the address.
  - Indic state names are often translations: পশ্চিমবঙ্গ = West Bengal.
- **Noisy addresses.**
  - Components are reordered.
  - Abbreviations differ by source (S2 upper-cases US addresses; S3 uses full US state names and Indian state codes).
  - Null tokens appear inside addresses, and 3% of addresses are empty.
  - Landmark phrases (Near/Opp/Behind) are common in India.
  - House numbers are truncated or padded (3424→424, 203→0203).
  - Postcodes are almost never present.
- **Test shift.** Test has more S2/S3 records per S1 (5.5–5.8 vs 4.7) and a new country, France (15% of S1s), with no labels.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted pair classifier + assignment-aware decision layer (hybrid).
**Core Innovation:**

- A cross-script phonetic key: one consonant skeleton across 9 Indic scripts and Latin spelling variants.
- Soft house-number features (prefix/suffix containment, shift size) that separate truncation noise from shifted-number decoys.
- Sibling features that compare a candidate with the S1's other candidates, which is where a decoy's real twin usually sits.
- A decision layer that enforces the one-S1-per-record partition and tunes thresholds on a stable plateau of the exact metric.

---

## 3. Candidate Generation (Blocking)
Every blocker runs inside one country; there are no cross-country pairs. Records are first normalised:

- **Name.** Core name and legal form are kept separately, with an alias for f/k/a and t/a; legal forms written in Indic scripts are detected.
- **Address.** One schema for all countries: state, city, locality, street, building, premise numbers, unit, floor, landmark and district. India-specific parsing covers sector/phase/block, nagar/colony, landmarks and typed plot/flat/khasra numbers.

**Blocking keys used (union of 13):**
- Top-K TF-IDF, per (country, state group) and per source:
  - char-3-gram TF-IDF on the transliterated core name (K=15);
  - 3–4-gram TF-IDF on the phonetic skeleton (K=3);
  - char TF-IDF on the street-level address alone (K=10), for renamed businesses;
  - name TF-IDF against only the records with no parseable place (K=2).
- Exact keys (hashed, posting-capped):
  - (place, rare name token);
  - (place, phonetic name);
  - (place, house number, street);
  - (place, premise number), filtered by a loose vectorised name similarity;
  - India (place, locality, phonetic name) and (place, locality, premise number);
  - a placeless rare token;
  - state-level safety nets;
  - S3 aliases as extra names.
- Every key that contains a number has a number-free backup key.
- **Candidate pairs generated:**
  - train: 156.8M (71 per S1; India 94, US 56);
  - test: 145.7M (84 per S1; France 127, India 95, US 54).
- **How we ensured true matches were not lost:**
  - K values were tuned from one run that kept every candidate's rank, taking the knee of the recall vs candidates Pareto front.
  - Recall was measured by country, by source and for each of the 21 Indian metros.
  - Missed pairs were dumped with a likely reason.
  - Full-train pair recall is **0.9813** (India 0.9719, US 0.9876). 91.7% of India and 95.9% of US non-singleton S1s have their complete match set among the candidates.

---

## 4. Matching Model

**Features used (105, float32, streamed per partition):**
- **Name** (rapidfuzz, on the transliterated core name):
  - ratio, token_set, token_sort, partial ratio and Jaro-Winkler; phonetic-skeleton ratio; best S3-alias token_set;
  - char-TF-IDF cosine; IDF-weighted token overlap;
  - extra-token count, IDF sum and maximum IDF (the "+ Industries" decoys);
  - legal-form relation (same / subset / overlap / conflict / missing);
  - domain, handle, initials, Indic-script and junk-prefix flags;
  - exact core-name equality, and how many S1s and targets share the core name.
- **Address:**
  - street ratio and token_set; street-address token_set and char-TF-IDF cosine;
  - soft house and premise number features: exact, prefix/suffix containment, digit edit distance, log absolute difference, shift of 1–10, one side missing;
  - unit equal/conflict;
  - overlaps of locality phonetic keys, building, street key, premise sets, places and landmarks;
  - S1 city among the target's places; state and state-group agreement; empty-address and placeless flags.
- **Context** (so the model stays robust to test having more candidates per S1):
  - a base score, with its rank, gap to the best and z-score within the S1's candidates;
  - name and address ranks and gaps;
  - reverse rank and gap among the S1s that share the target, and "this S1 is the target's best";
  - candidate counts; which of the 13 blockers fired and the blocker ranks; source S2/S3.
- **Sibling features:** the gap to the S1's best candidate on address cosine, name, premise, street and exact number, and a "a sibling has the exact number and a similar name, this pair doesn't" decoy flag.
- Raw country is **not** a feature. No pair is rejected by a hard rule; conflicts are features.

**Model type:** LightGBM binary classifier.
- Training rows: every positive, the hard negatives (base-score rank ≤ 10) and a 5% sample of easy negatives (weight 20), from a random 30% of train S1s (662k S1s, 9.0M rows).
- 5-fold GroupKFold by S1 on a fixed fold assignment, with early stopping.
- Out-of-fold predictions for **all** 157M train pairs: each S1 is scored by the model of its own fold.
- Test probabilities are the mean of the 5 fold models.

**Threshold selection method:**
- On OOF with the exact macro F0.5:
  1. Assignment: each S2/S3 record is kept only for the S1 with the highest probability.
  2. Per S1, keep candidates with p ≥ t_keep and p ≥ ratio × best (several per source allowed), capped at 11.
  3. Predict empty when best < t_null.
- The grid covers t_keep, ratio, t_null and assignment on/off. We chose the neighbourhood-smoothed optimum (a flat plateau) rather than the single peak: **t_keep = 0.7, ratio = 0.8, t_null = 0.7, assignment on, cap 11**. The single peak was 0.97428 and the chosen plateau point scores 0.97411.
- Thresholds tuned on the US transfer to India unchanged.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, OOF on full train):** **0.9741** (US 0.9787, India 0.9672). A flat 0.5 cutoff scores 0.9669. Singletons go from 0.896 with the flat cutoff to 0.959 (from 0.907 with the flat cutoff) with the decision layer.
- **Iteration log:** 
  | step | CV macro F0.5 |
  |---|---|
  | model + flat 0.5 cutoff | 0.9618 |
  | + decision layer (assignment + stable thresholds) | 0.9712 |
  | + name-commonness and sibling (decoy-twin) features (Phase 4, iteration 1) | **0.9741** |

  Iteration 1 cut decoy false merges from 56.7k to 33.3k and generic-name merges from 14.2k to 5.3k. Phase 4 then stopped, because the gain (+0.00295) was below the 0.003 bar. The remaining oracle gains (the macro F0.5 if a category's errors were fixed):

  | error category | oracle gain |
  |---|---|
  | blocking misses | 0.0070 |
  | empty-address misses | 0.0057 |
  | other low-probability misses | 0.0057 |
  | decoy false merges | 0.0044 |
  | wrong-S1 merges | 0.0015 |
- **Common false positives (wrong merges):**
  - planted decoys: the same name ± one word, a house number shifted by a few, a different legal form;
  - generic names of different S1s in the same city.
- **Common false negatives (missed matches):**
  - blocking misses (1.9% of true pairs);
  - empty-address targets with lightly typo'd names;
  - renamed businesses where only the address agrees;
  - Indic-script names with noisy addresses.
- **France proxy:** a model trained on one country and applied to the other scores 0.852 (US→India) and 0.958 (India→US), so French performance is expected to be below the in-country CV. The model uses no country feature; the test model is trained on US+India.

---

## 6. Conclusion
Recall-first blocking (0.981 pair recall) followed by a feature-rich LightGBM pair model and an assignment-aware decision layer reaches 0.974 macro F0.5 out-of-fold. The largest gains came from three places:

- treating the one-S1-per-record partition explicitly (+0.009);
- tuning thresholds on the exact metric, which matters most for singletons;
- features that compare a candidate with its S1's other candidates and measure how common a name is, which handle the planted decoys.

Next steps are France-specific normalisation (France has no labels), the remaining blocking misses in India, and name-only (empty-address) matches.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source in `src/ber/`, entry point `run_pipeline.py` (`train` / `predict`),
pinned `requirements.txt`, and `README.md` with exact commands and timings. Stage modules: `prep` (normalisation
+ parsing + transliteration, cached by code hash) → `blocking` → `features` → `train` → `decide` → `predict`.
`python run_pipeline.py predict` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and
runs the official validator.

### B. Additional Results
**Blocking recall (full train, K name 15 / addr 10 / phon 3 / placeless 2):**

| segment | pair recall |
|---|---|
| all | 0.9813 |
| US | 0.9876 |
| India | 0.9719 |
| S2 / S3 (India) | 0.9746 / 0.9694 |
| S2 / S3 (US) | 0.9881 / 0.9871 |

India metros range from 0.959 (Patna) to 0.982 (Ernakulam), with Mumbai at 0.965. Candidates per S1: train 71, test 84 (France 127).

**OOF macro F0.5 by true match-set size** (decision layer vs flat 0.5 cutoff):

| size | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| decision layer | 0.959 | 0.934 | 0.970 | 0.977 | 0.980 | 0.981 | 0.982 | 0.983 | 0.983 |
| flat 0.5 | 0.907 | 0.925 | 0.961 | 0.971 | 0.976 | 0.979 | 0.981 | 0.982 | 0.983 |

**Top features by gain:** base score, house-number log difference, reverse rank among the S1s sharing the target, gap to that target's best S1, within-S1 z-score, state-group agreement, legal-form relation, max IDF of extra target tokens, address cosine, name commonness, and the address gap to the best sibling. The full list is in `reports/feature_importance.md`.

**Test prediction sanity check:**
- Empty predictions: 5.6–5.7% in every country (train singleton rate 5.6%).
- Mean match-set size: France 3.20, India 3.25, US 3.33.
- The official validator passes.

**No external data, APIs, geocoding or business-registry lookups were used.** The geography lists (state names,
French departments, old/new Indian city names) and the transliteration tables are hand-written general knowledge.
