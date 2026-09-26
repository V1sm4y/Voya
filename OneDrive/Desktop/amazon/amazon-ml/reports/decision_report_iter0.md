# Decision layer (OOF, full train, exact macro F0.5)

flat 0.5 cutoff (no assignment): **0.9618**

single peak: {'t_keep': 0.7, 'ratio': 0.0, 't_null': 0.7, 'assign': True, 'f05': 0.9712398578951857}

chosen (stable, neighbourhood-smoothed): {'t_keep': 0.6, 'ratio': 0.4, 't_null': 0.6, 'assign': True, 'f05': 0.9711526236034284, '_t_keep': 4, '_ratio': 2, '_t_null': 3, 'f05_smooth': 0.9709428556884738}

## top 15 settings

|   t_keep |   ratio |   t_null | assign   |    f05 |
|---------:|--------:|---------:|:---------|-------:|
|      0.7 |     0   |      0.7 | True     | 0.9712 |
|      0.7 |     0.6 |      0.7 | True     | 0.9712 |
|      0.7 |     0.4 |      0.7 | True     | 0.9712 |
|      0.7 |     0.2 |      0.7 | True     | 0.9712 |
|      0.3 |     0.6 |      0.7 | True     | 0.9712 |
|      0.4 |     0.6 |      0.7 | True     | 0.9712 |
|      0.2 |     0.6 |      0.7 | True     | 0.9712 |
|      0.5 |     0.6 |      0.7 | True     | 0.9712 |
|      0.4 |     0.6 |      0.6 | True     | 0.9712 |
|      0.3 |     0.6 |      0.6 | True     | 0.9712 |
|      0.2 |     0.6 |      0.6 | True     | 0.9712 |
|      0.5 |     0.6 |      0.6 | True     | 0.9712 |
|      0.6 |     0   |      0.7 | True     | 0.9712 |
|      0.6 |     0.4 |      0.7 | True     | 0.9712 |
|      0.6 |     0.6 |      0.7 | True     | 0.9712 |

## Threshold transfer / cross-country (France proxy)

| thresholds tuned on   | evaluated on   | tuned                                   |   F0.5 (in-country model, transferred thresholds) |   F0.5 (other-country model + thresholds = France proxy) |   F0.5 (global chosen thresholds) |
|:----------------------|:---------------|:----------------------------------------|--------------------------------------------------:|---------------------------------------------------------:|----------------------------------:|
| US                    | India          | keep 0.6 ratio 0.4 null 0.6 assign True |                                            0.9651 |                                                   0.8535 |                            0.9651 |
| India                 | US             | keep 0.7 ratio 0.2 null 0.7 assign True |                                            0.9752 |                                                   0.9545 |                            0.9752 |

## By country

| country   |   chosen |   flat_0.5 |
|:----------|---------:|-----------:|
| India     |   0.9651 |     0.9553 |
| US        |   0.9752 |     0.9661 |

## By true match-set size (8 = 8+)

|   size |   ('chosen', 'mean') |   ('chosen', 'size') |   ('flat_0.5', 'mean') |   ('flat_0.5', 'size') |
|-------:|---------------------:|---------------------:|-----------------------:|-----------------------:|
|      0 |               0.9445 |               123247 |                 0.8959 |                 123247 |
|      1 |               0.9261 |               119157 |                 0.9139 |                 119157 |
|      2 |               0.9658 |               375212 |                 0.955  |                 375212 |
|      3 |               0.9744 |               530841 |                 0.9666 |                 530841 |
|      4 |               0.978  |               484115 |                 0.9722 |                 484115 |
|      5 |               0.9803 |               321957 |                 0.9758 |                 321957 |
|      6 |               0.9813 |               164868 |                 0.978  |                 164868 |
|      7 |               0.9825 |                63968 |                 0.98   |                  63968 |
|      8 |               0.983  |                23456 |                 0.9811 |                  23456 |

## Singletons vs non-singletons

| kind          |   chosen |   flat_0.5 |
|:--------------|---------:|-----------:|
| non-singleton |   0.9727 |     0.9657 |
| singleton     |   0.9445 |     0.8959 |


exactness check: metrics.report 0.971153 vs fast 0.971153

saved decision config: {'t_keep': 0.6, 'ratio': 0.4, 't_null': 0.6, 'assign': True, 'cap': 11, 'cv_macro_f05': 0.9711526236034284, 'cv_by_country': {'India': 0.9651, 'US': 0.9752}}
