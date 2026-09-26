# Blocking — train_dev

config: Config(k_name=10, k_addr=10, k_alias=1, k_phon=3, k_placeless=2, k_max=0, premise_cap=1000, state_cap=25, placeless_cap=20, state_rare_df=500, premise_name_sim=50, premise_phon_sim=60, tfidf_threshold=0.1, tfidf_max_df=0.05, rare_df=2000, rare_per_s1=2, cap=300, threads=9)

S1 133,463 | targets 10,305,310 | generation 14.2 min

## Candidates per S1

| country   |   count |   mean |   std |   min |   50% |   90% |   99% |   max |
|:----------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| India     |   35653 |   67.2 |  52.6 |    20 |    52 |   100 |   335 |  1191 |
| US        |   77810 |   47.7 |  17.4 |    21 |    44 |    59 |   124 |   341 |
| ALL       |  113463 |   53.8 |  34   |    20 |    45 |    69 |   217 |  1191 |

total candidate pairs: 6,108,079 (S2 2,952,749, S3 3,155,330)

## Pair recall

overall: **0.9819** of 392,625 true pairs (outside pool: 0.0000%)

| country   |   true_pairs |   recall |
|:----------|-------------:|---------:|
| India     |       123670 |   0.9729 |
| US        |       268955 |   0.9861 |

|                 |   true_pairs |   recall |
|:----------------|-------------:|---------:|
| ('India', 'S2') |        59918 |   0.9762 |
| ('India', 'S3') |        63752 |   0.9697 |
| ('US', 'S2')    |       129908 |   0.986  |
| ('US', 'S3')    |       139047 |   0.9863 |

## S1 with the full match set covered (non-singletons)

| country   |   full_cover |
|:----------|-------------:|
| India     |       0.9181 |
| US        |       0.9542 |

## Blockers (at the selected K)

| blocker         |            pairs |   recall_alone |   unique_catches |   unique_share_of_true |
|:----------------|-----------------:|---------------:|-----------------:|-----------------------:|
| name_tfidf      |      2.26916e+06 |         0.743  |             2902 |                 0.0074 |
| rare_token      | 715534           |         0.4127 |              318 |                 0.0008 |
| phonetic_city   | 302069           |         0.4614 |              494 |                 0.0013 |
| alias           | 119656           |         0.0184 |               49 |                 0.0001 |
| addr_key        | 203817           |         0.3702 |                6 |                 0      |
| addr_tfidf      |      2.26292e+06 |         0.8326 |            15687 |                 0.04   |
| in_loc_name     |  53041           |         0.115  |                0 |                 0      |
| in_loc_premise  | 302667           |         0.1342 |               63 |                 0.0002 |
| name_phon_tfidf | 679782           |         0.5057 |              159 |                 0.0004 |
| placeless_name  | 453820           |         0.037  |               75 |                 0.0002 |
| place_premise   | 477319           |         0.5269 |             2612 |                 0.0067 |
| placeless_token | 247062           |         0.0201 |              325 |                 0.0008 |
| state_keys      | 834837           |         0.7685 |              257 |                 0.0007 |

## India metros (probe: random S1s per metro outside DEV; pool = same state groups)

| city                |   true_pairs |   recall |   in_probe |   cands_per_s1 |
|:--------------------|-------------:|---------:|-----------:|---------------:|
| patna               |         3455 |   0.9589 |          1 |        60.228  |
| chennai             |         3443 |   0.9599 |          1 |        79.395  |
| gautam buddha nagar |         3435 |   0.9642 |          1 |        79.507  |
| mumbai              |         3391 |   0.9643 |          1 |       109.815  |
| new delhi           |         3433 |   0.9656 |          1 |       110.048  |
| east delhi          |         3477 |   0.9661 |          1 |       124.886  |
| south delhi         |         3472 |   0.9675 |          1 |       121.541  |
| pune                |         3513 |   0.9681 |          1 |        84.423  |
| lucknow             |         3436 |   0.9683 |          1 |        57.757  |
| ahmedabad           |         3414 |   0.9692 |          1 |        73.432  |
| howrah              |         3398 |   0.9706 |          1 |       142.447  |
| thane               |         3475 |   0.9715 |          1 |       123.23   |
| west delhi          |         3528 |   0.9719 |          1 |       150.039  |
| ghaziabad           |         3514 |   0.9732 |          1 |        60.791  |
| jaipur              |        55074 |   0.975  |          0 |        63.6245 |
| bengaluru           |         3458 |   0.9754 |          1 |        87.004  |
| south west delhi    |         3527 |   0.9782 |          1 |       173.746  |
| gurugram            |         3510 |   0.9786 |          1 |        65.183  |
| hyderabad           |         3502 |   0.9817 |          1 |        96.134  |
| ernakulam           |         3436 |   0.9831 |          1 |        60.796  |
| kolkata             |         3446 |   0.9864 |          1 |        85.209  |

all India-metro pairs: recall 0.9728

## Missed true pairs by likely reason

| reason                                           |   India |   US |   share_of_missed |
|:-------------------------------------------------|--------:|-----:|------------------:|
| empty target address, similar name (ranked out)  |     376 | 1428 |             0.254 |
| similar name + shared place, ranked out / capped |     665 |  706 |             0.193 |
| similar address, ranked out                      |     442 |  365 |             0.114 |
| Indic-script name + noisy address                |     692 |    0 |             0.098 |
| unrelated name, address similar but ranked out   |     329 |  348 |             0.096 |
| noisy name + noisy address                       |     385 |  142 |             0.074 |
| empty target address + noisy name                |     156 |  315 |             0.066 |
| similar name, no shared place                    |      57 |  346 |             0.057 |
| unrelated name + noisy address                   |     231 |   60 |             0.041 |
| different state/region                           |      24 |   22 |             0.006 |
