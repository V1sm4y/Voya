# Experiments log

Scores are macro F0.5 unless stated. "DEV" = geographic DEV (biased easy for India, so use it only for
comparisons). "CV" = out-of-fold on full train (5 folds grouped by S1). Newest entries at the bottom.

| date | phase | change | metric | before | after | notes |
|---|---|---|---|---|---|---|
| 2026-09-26 | baseline | all-empty | DEV F0.5 | – | 0.0559 | = singleton share |
| 2026-09-26 | baseline | exact core_name + same city | DEV F0.5 | – | 0.5762 (w 0.5701) | old random DEV: 0.5432 |
| 2026-09-26 | baseline | + no legal-form conflict (hard rule, baseline only) | DEV F0.5 | 0.5762 | 0.5984 | +0.020 on both DEVs |
| 2026-09-26 | blocking | 13 blockers, K name10/addr10/phon3/placeless2 | pair recall (full train) | – | 0.9798 @ 62 cands/S1 | DEV 0.9819 @ 54 |
| 2026-09-26 | 0 blocking | K name 10 -> 15; hashed exact keys (memory fix) | pair recall (full train) | 0.9798 @ 62 | **0.9813 @ 71** | India 0.9719, US 0.9876; peak RSS 8.6 GB (was ~26 GB footprint) |
| 2026-09-26 | 1 features | features.py: 88 features, partition-streamed | throughput | 13k pairs/s (tiny partitions) | 57k pairs/s (1.5M-pair partition) | positives vs negatives separate on every name/address/context feature (smoke: 5k S1) |
| 2026-09-26 | 2 model | LightGBM 88 feats, 30% S1 train rows (pos + hard rank<=10 + 5% easy, w=20) | OOF PR-AUC | – | 0.9967 (India 0.9955 / US 0.9973) | cross-country: US->India 0.948, India->US 0.990 |
| 2026-09-26 | 3 decision | flat 0.5 -> assignment + keep 0.6 / ratio 0.4 / null 0.6 (stable plateau) | CV macro F0.5 | 0.9618 | **0.9712** | India 0.9651, US 0.9752; singletons 0.896 -> 0.956; peak 0.97124 at 0.7/0/0.7 |
| 2026-09-26 | 4 errors (iter 0) | OOF error analysis at 0.9712 | oracle gains | – | – | blocking miss 0.0069, decoy false merge 0.0069, empty-address miss 0.0058, other low-p 0.0032, wrong-S1 merge 0.0032, generic-name 0.0016 |
| 2026-09-26 | 4 iter 1 | + name-commonness (core_eq, s/t core freq) + sibling-relative features (gap to S1's best candidate on a_cos/name/premise/street/num_eq, exact-number-twin decoy flag, extra-token gap); 88 -> 105 feats | CV macro F0.5 | 0.9712 | **0.9741** (+0.0030) | India 0.9651->0.9672, US 0.9752->0.9787; PR-AUC 0.9967->0.9974; decoy false merges 56.7k->33.3k, generic-name merges 14.2k->5.3k, wrong-S1 28.5k->12.9k; thresholds keep 0.7/ratio 0.8/null 0.7 |
| 2026-09-26 | 4 stop | iteration gain +0.00295 < 0.003 bar -> Phase 4 stopped after 1 iteration (rule from brief) | – | – | – | next candidates: blocking misses (oracle 0.0070), empty-address misses (0.0057), low-p misses (0.0057) |
| 2026-09-26 | 5 test | full test inference, 5-fold mean, decision keep 0.7/ratio 0.8/null 0.7 | predicted sizes | – | mean 3.20 (FR) / 3.25 (IN) / 3.33 (US); empty 5.7% / 5.6% / 5.7% | validator PASS; France distribution tracks US/India (sanity) |
