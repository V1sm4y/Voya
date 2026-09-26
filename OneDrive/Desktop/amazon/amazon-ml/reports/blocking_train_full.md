# Blocking — train_full

config: Config(k_name=15, k_addr=10, k_alias=1, k_phon=3, k_placeless=2, k_max=0, premise_cap=1000, state_cap=25, placeless_cap=20, state_rare_df=500, premise_name_sim=50, premise_phon_sim=60, tfidf_threshold=0.1, tfidf_max_df=0.05, rare_df=2000, rare_per_s1=2, cap=300, threads=9)

S1 2,206,821 | targets 10,320,219 | generation 36.0 min

## Candidates per S1

| country   |            count |   mean |   std |   min |   50% |   90% |   99% |   max |
|:----------|-----------------:|-------:|------:|------:|------:|------:|------:|------:|
| India     | 883188           |   93.8 |  76.5 |    20 |    66 |   167 |   421 |  2916 |
| US        |      1.32363e+06 |   55.8 |  16.9 |    20 |    52 |    65 |   114 |   367 |
| ALL       |      2.20682e+06 |   71   |  53.5 |    20 |    55 |   101 |   327 |  2916 |

total candidate pairs: 156,754,117 (S2 74,838,722, S3 81,915,395)

## Pair recall

overall: **0.9813** of 7,638,365 true pairs (outside pool: 0.0000%)

| country   |   true_pairs |   recall |
|:----------|-------------:|---------:|
| India     |  3.05984e+06 |   0.9719 |
| US        |  4.57852e+06 |   0.9876 |

|                 |   true_pairs |   recall |
|:----------------|-------------:|---------:|
| ('India', 'S2') |  1.48054e+06 |   0.9746 |
| ('India', 'S3') |  1.5793e+06  |   0.9694 |
| ('US', 'S2')    |  2.21307e+06 |   0.9881 |
| ('US', 'S3')    |  2.36545e+06 |   0.9871 |

## S1 with the full match set covered (non-singletons)

| country   |   full_cover |
|:----------|-------------:|
| India     |       0.9167 |
| US        |       0.9586 |

## Blockers (at the selected K)

| blocker         |       pairs |   recall_alone |   unique_catches |   unique_share_of_true |
|:----------------|------------:|---------------:|-----------------:|-----------------------:|
| name_tfidf      | 6.62024e+07 |         0.7531 |            68328 |                 0.0089 |
| rare_token      | 1.58046e+07 |         0.4077 |             6693 |                 0.0009 |
| phonetic_city   | 8.70013e+06 |         0.4603 |            14359 |                 0.0019 |
| alias           | 2.41933e+06 |         0.0179 |             1018 |                 0.0001 |
| addr_key        | 4.05751e+06 |         0.3342 |              189 |                 0      |
| addr_tfidf      | 4.40191e+07 |         0.821  |           278317 |                 0.0364 |
| in_loc_name     | 1.94537e+06 |         0.1505 |               26 |                 0      |
| in_loc_premise  | 1.79113e+07 |         0.1848 |             2678 |                 0.0004 |
| name_phon_tfidf | 1.32213e+07 |         0.4851 |             2459 |                 0.0003 |
| placeless_name  | 8.8269e+06  |         0.0402 |              991 |                 0.0001 |
| place_premise   | 1.23381e+07 |         0.5286 |            50754 |                 0.0066 |
| placeless_token | 4.92064e+06 |         0.0224 |             6137 |                 0.0008 |
| state_keys      | 1.60501e+07 |         0.7324 |             4295 |                 0.0006 |

## India metros (probe: random S1s per metro outside DEV; pool = same state groups)

| city                |   true_pairs |   recall |   in_probe |   cands_per_s1 |
|:--------------------|-------------:|---------:|-----------:|---------------:|
| patna               |        32305 |   0.9602 |          0 |        68.4907 |
| gautam buddha nagar |        43010 |   0.9632 |          0 |        92.6297 |
| mumbai              |       259488 |   0.9665 |          0 |       118.742  |
| south delhi         |        74163 |   0.9674 |          0 |       127.972  |
| new delhi           |        91016 |   0.9675 |          0 |       113.48   |
| howrah              |        89855 |   0.969  |          0 |       145.779  |
| chennai             |        92982 |   0.97   |          0 |        88.1401 |
| east delhi          |        45111 |   0.9711 |          0 |       136.588  |
| west delhi          |        42596 |   0.9723 |          0 |       153.095  |
| south west delhi    |        33015 |   0.9729 |          0 |       178.724  |
| gurugram            |        58214 |   0.9736 |          0 |        74.1677 |
| lucknow             |        43328 |   0.9736 |          0 |        67.384  |
| pune                |       134935 |   0.974  |          0 |        92.9254 |
| bengaluru           |       187519 |   0.9742 |          0 |        95.0291 |
| ghaziabad           |        31897 |   0.9744 |          0 |        67.5333 |
| thane               |        81327 |   0.9745 |          0 |       126.053  |
| ahmedabad           |        76623 |   0.9759 |          0 |        83.6497 |
| jaipur              |        55074 |   0.9761 |          0 |        72.5644 |
| hyderabad           |       130198 |   0.979  |          0 |       105.626  |
| kolkata             |        43379 |   0.9829 |          0 |        96.7635 |
| ernakulam           |        31411 |   0.9832 |          0 |        68.6824 |

all India-metro pairs: recall 0.9720

## Missed true pairs by likely reason

| reason                                           |   India |    US |   share_of_missed |
|:-------------------------------------------------|--------:|------:|------------------:|
| empty target address, similar name (ranked out)  |    8702 | 19708 |             0.199 |
| similar name + shared place, ranked out / capped |   17259 |  9895 |             0.19  |
| similar address, ranked out                      |   14062 |  5647 |             0.138 |
| Indic-script name + noisy address                |   16086 |     0 |             0.113 |
| unrelated name, address similar but ranked out   |    9428 |  5820 |             0.107 |
| noisy name + noisy address                       |    8517 |  1525 |             0.07  |
| empty target address + noisy name                |    4199 |  4909 |             0.064 |
| similar name, no shared place                    |    1660 |  5716 |             0.052 |
| unrelated name + noisy address                   |    5239 |   648 |             0.041 |
| different state/region                           |     747 |  2797 |             0.025 |
