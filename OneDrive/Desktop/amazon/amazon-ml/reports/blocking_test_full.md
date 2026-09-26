# Blocking — test_full

config: Config(k_name=15, k_addr=10, k_alias=1, k_phon=3, k_placeless=2, k_max=0, premise_cap=1000, state_cap=25, placeless_cap=20, state_rare_df=500, premise_name_sim=50, premise_phon_sim=60, tfidf_threshold=0.1, tfidf_max_df=0.05, rare_df=2000, rare_per_s1=2, cap=300, threads=9)

S1 1,732,544 | targets 9,969,589 | generation 37.2 min

## Candidates per S1

| country   |            count |   mean |   std |   min |   50% |   90% |   99% |   max |
|:----------|-----------------:|-------:|------:|------:|------:|------:|------:|------:|
| France    | 259452           |  126.8 |  73.7 |    20 |   104 |   229 |   370 |   653 |
| India     | 809986           |   95.3 |  79.8 |    20 |    66 |   172 |   436 |  1967 |
| US        | 663106           |   53.6 |  15.5 |    20 |    51 |    61 |   102 |   357 |
| ALL       |      1.73254e+06 |   84.1 |  67.6 |    20 |    58 |   153 |   369 |  1967 |

total candidate pairs: 145,655,593 (S2 69,597,730, S3 76,057,863)
