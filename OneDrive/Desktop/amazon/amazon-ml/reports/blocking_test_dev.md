# Blocking — test_dev

config: Config(k_name=10, k_addr=10, k_alias=1, k_phon=3, k_placeless=2, k_max=0, premise_cap=1000, state_cap=25, placeless_cap=20, state_rare_df=500, premise_name_sim=50, premise_phon_sim=60, tfidf_threshold=0.1, tfidf_max_df=0.05, rare_df=2000, rare_per_s1=2, cap=300, threads=9)

S1 85,000 | targets 9,937,029 | generation 14.1 min

## Candidates per S1

| country   |   count |   mean |   std |   min |   50% |   90% |   99% |   max |
|:----------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| France    |    7833 |  116.6 |  66.3 |    28 |   100 |   201 | 360.7 |   516 |
| India     |   38494 |   85.5 |  90.5 |    20 |    52 |   182 | 470.1 |  1378 |
| US        |   38673 |   45.3 |  19.5 |    27 |    42 |    54 |  93.3 |   327 |
| ALL       |   85000 |   70.1 |  69.8 |    20 |    47 |   126 | 383   |  1378 |

total candidate pairs: 5,957,998 (S2 2,793,738, S3 3,164,260)
