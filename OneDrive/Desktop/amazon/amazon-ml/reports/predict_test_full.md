# Prediction — test_full

decision config: {'t_keep': 0.7, 'ratio': 0.8, 't_null': 0.7, 'assign': True, 'cap': 11, 'cv_macro_f05': 0.9741057723544665, 'cv_by_country': {'India': 0.9672, 'US': 0.9787}}

matching: {'path': '/Users/tanush.s.vashisht/Amazon ML/output/matching_results.tsv', 'rows': 1732544, 'empty': 97624, 'links': 5671039}

candidate: {'path': '/Users/tanush.s.vashisht/Amazon ML/output/candidate_pairs.tsv', 'rows': 1732544, 'empty': 0, 'links': 145655593}

mean predicted match-set size by country: {'France': 3.199, 'India': 3.253, 'US': 3.327}

## Predicted match-set size distribution (share of S1; 8 = 8+)

| country   |      0 |      1 |      2 |      3 |      4 |      5 |      6 |      7 |      8 |
|:----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| France    | 0.0567 | 0.0891 | 0.1967 | 0.2455 | 0.2031 | 0.123  | 0.0576 | 0.0207 | 0.0076 |
| India     | 0.0561 | 0.0855 | 0.1911 | 0.2408 | 0.2052 | 0.1282 | 0.0615 | 0.0232 | 0.0085 |
| US        | 0.0565 | 0.0714 | 0.1839 | 0.2436 | 0.2101 | 0.1347 | 0.0658 | 0.0249 | 0.009  |

validator exit code: 0
```
ML Challenge 2026 — submission validator
  test dir: /Users/tanush.s.vashisht/Amazon ML/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (97624 empty, 1634920 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

WARNING: ID-existence check is OFF (the default) — not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
PASS — no blocking issues found. Safe to submit.

```
