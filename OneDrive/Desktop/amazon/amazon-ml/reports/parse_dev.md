# Parse report — train/dev

[train/dev] S1 113,463 | S2 5,027,411 | S3 5,277,899 | countries ['India', 'US']

## 1. Field coverage (share of records with the field filled)

|                 |      n |   state |   city |   locality |   street |   building |   premise_no |   unit |   floor |   landmark |   admin |   pin |
|:----------------|-------:|--------:|-------:|-----------:|---------:|-----------:|-------------:|-------:|--------:|-----------:|--------:|------:|
| ('S1', 'India') |  35653 |   100   |  100   |       88.4 |     34.1 |       18   |         80.1 |    0   |    10.9 |       16.4 |     3.3 |   0.2 |
| ('S1', 'US')    |  77810 |   100   |  100   |        1.1 |     97.1 |        0.2 |         99.6 |    7.6 |     0.9 |        0   |     0   |   0   |
| ('S2', 'India') | 100000 |    96.8 |   94.4 |       79.6 |     26.3 |       20.1 |         82.5 |    0   |    15   |       11.4 |     1.8 |   0.9 |
| ('S2', 'US')    | 100000 |    96.3 |   93.8 |        3.7 |     92.7 |        0.8 |         82   |    6.8 |     0   |        0   |     0   |   0   |
| ('S3', 'India') | 100000 |    96.6 |   96.1 |       77   |     20.3 |       15.8 |         80.9 |    0   |    12   |        8.6 |     1.2 |   0.8 |
| ('S3', 'US')    | 100000 |    96.5 |   93.9 |        3.7 |     93   |        1   |         82.8 |   15.6 |     1.3 |        0   |     0.1 |   0   |

## 2. Field agreement on exact-name + same-city pairs (row % within pair kind)

### India  (pairs: TP 43,439, FP_other 18,987, FP_decoy 8,799)

**premise_no**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     7.4 |       72.6 |      20   |
| FP_other |     0.3 |       74.8 |      24.8 |
| TP       |    63.8 |       12.7 |      23.5 |

**locality_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |    29.2 |       43.9 |      26.8 |
| FP_other |     2.2 |       64.1 |      33.7 |
| TP       |    70   |        7.7 |      22.3 |

**street_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |    10.8 |        8.2 |      81   |
| FP_other |     0.2 |       12.4 |      87.5 |
| TP       |    26.9 |        0.5 |      72.5 |

**building_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     5.2 |        1.9 |      92.9 |
| FP_other |     0   |        2.9 |      97.1 |
| TP       |    14.1 |        0.5 |      85.4 |

**landmark**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     4.7 |        1.4 |      93.9 |
| FP_other |     0   |        1.7 |      98.3 |
| TP       |    12.3 |        0.2 |      87.5 |

**unit**

| kind     |   missing |
|:---------|----------:|
| FP_decoy |       100 |
| FP_other |       100 |
| TP       |       100 |

**legal**

| kind     |   conflict |   missing |   overlap |   same |   subset |
|:---------|-----------:|----------:|----------:|-------:|---------:|
| FP_decoy |       23.7 |      18.4 |      13.9 |   22.1 |     21.9 |
| FP_other |       12.9 |      24.6 |       0.9 |   45.8 |     15.8 |
| TP       |        0   |      25.1 |       0   |   74.8 |      0.2 |

### US  (pairs: TP 116,949, FP_decoy 20,365, FP_other 7,931)

**premise_no**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     2.4 |       89.7 |       7.8 |
| FP_other |     0   |       82.5 |      17.5 |
| TP       |    68.3 |       14.9 |      16.8 |

**locality_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     0.1 |        0.1 |      99.8 |
| FP_other |     0   |        0   |     100   |
| TP       |     0.2 |        0   |      99.8 |

**street_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |    77.6 |       19.5 |       3   |
| FP_other |     0.1 |       97.4 |       2.6 |
| TP       |    84.7 |       11.7 |       3.6 |

**building_key**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     0.1 |          0 |      99.9 |
| FP_other |     0   |          0 |     100   |
| TP       |     0.1 |          0 |      99.9 |

**landmark**

| kind     |   missing |
|:---------|----------:|
| FP_decoy |       100 |
| FP_other |       100 |
| TP       |       100 |

**unit**

| kind     |   agree |   conflict |   missing |
|:---------|--------:|-----------:|----------:|
| FP_decoy |     2.6 |        0.4 |      96.9 |
| FP_other |     0   |        0.2 |      99.8 |
| TP       |     3   |        0.4 |      96.6 |

**legal**

| kind     |   conflict |   missing |   overlap |   same |   subset |
|:---------|-----------:|----------:|----------:|-------:|---------:|
| FP_decoy |       49.5 |      46.8 |       0.3 |    1.9 |      1.6 |
| FP_other |       20   |      68.7 |       0   |   11.3 |      0   |
| TP       |        0.9 |      56.4 |       0   |   42.6 |      0.1 |

## 3. Rule variants (macro F0.5 on this DEV)

| variant                                                                            |   macro_f05 |   macro_f05_w |   link_precision |   link_recall |   F0.5 India |   F0.5 US |   prec India |   prec US |
|:-----------------------------------------------------------------------------------|------------:|--------------:|-----------------:|--------------:|-------------:|----------:|-------------:|----------:|
| baseline (name+city)                                                               |      0.5762 |        0.5701 |           0.7409 |        0.4085 |       0.523  |    0.6006 |       0.6099 |    0.8052 |
| + no legal conflict                                                                |      0.5984 |        0.5902 |           0.7998 |        0.4059 |       0.533  |    0.6284 |       0.6512 |    0.8746 |
| + no legal conflict + no premise/unit conflict                                     |      0.5639 |        0.5552 |           0.9434 |        0.3468 |       0.5076 |    0.5896 |       0.8634 |    0.9784 |
| + no legal conflict + no premise/unit conflict + India needs one address agreement |      0.561  |        0.5526 |           0.9766 |        0.3413 |       0.4985 |    0.5896 |       0.9715 |    0.9784 |
