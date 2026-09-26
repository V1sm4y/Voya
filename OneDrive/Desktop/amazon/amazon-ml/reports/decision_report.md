# Decision layer (OOF, full train, exact macro F0.5)

flat 0.5 cutoff (no assignment): **0.9669**

single peak: {'t_keep': 0.7, 'ratio': 0.4, 't_null': 0.7, 'assign': True, 'f05': 0.9742821405460445}

chosen (stable, neighbourhood-smoothed): {'t_keep': 0.7, 'ratio': 0.8, 't_null': 0.7, 'assign': True, 'f05': 0.9741057723544665, '_t_keep': 5, '_ratio': 4, '_t_null': 4, 'f05_smooth': 0.9739491528873571}

## top 15 settings

|   t_keep |   ratio |   t_null | assign   |    f05 |
|---------:|--------:|---------:|:---------|-------:|
|      0.7 |     0.4 |      0.7 | True     | 0.9743 |
|      0.7 |     0.2 |      0.7 | True     | 0.9743 |
|      0.7 |     0.6 |      0.7 | True     | 0.9743 |
|      0.7 |     0   |      0.7 | True     | 0.9743 |
|      0.7 |     0.2 |      0.8 | True     | 0.9742 |
|      0.7 |     0.4 |      0.8 | True     | 0.9742 |
|      0.7 |     0   |      0.8 | True     | 0.9742 |
|      0.7 |     0.6 |      0.8 | True     | 0.9742 |
|      0.5 |     0.8 |      0.7 | True     | 0.9741 |
|      0.3 |     0.8 |      0.7 | True     | 0.9741 |
|      0.2 |     0.8 |      0.7 | True     | 0.9741 |
|      0.4 |     0.8 |      0.7 | True     | 0.9741 |
|      0.6 |     0.8 |      0.7 | True     | 0.9741 |
|      0.7 |     0.8 |      0.7 | True     | 0.9741 |
|      0.6 |     0.8 |      0.8 | True     | 0.974  |

## Threshold transfer / cross-country (France proxy)

| thresholds tuned on   | evaluated on   | tuned                                   |   F0.5 (in-country model, transferred thresholds) |   F0.5 (other-country model + thresholds = France proxy) |   F0.5 (global chosen thresholds) |
|:----------------------|:---------------|:----------------------------------------|--------------------------------------------------:|---------------------------------------------------------:|----------------------------------:|
| US                    | India          | keep 0.7 ratio 0.8 null 0.7 assign True |                                            0.9672 |                                                   0.852  |                            0.9672 |
| India                 | US             | keep 0.7 ratio 0.0 null 0.7 assign True |                                            0.9788 |                                                   0.9583 |                            0.9787 |

## By country

| country   |   chosen |   flat_0.5 |
|:----------|---------:|-----------:|
| India     |   0.9672 |     0.9582 |
| US        |   0.9787 |     0.9726 |

## By true match-set size (8 = 8+)

|   size |   ('chosen', 'mean') |   ('chosen', 'size') |   ('flat_0.5', 'mean') |   ('flat_0.5', 'size') |
|-------:|---------------------:|---------------------:|-----------------------:|-----------------------:|
|      0 |               0.9591 |               123247 |                 0.9074 |                 123247 |
|      1 |               0.9338 |               119157 |                 0.9247 |                 119157 |
|      2 |               0.9697 |               375212 |                 0.9614 |                 375212 |
|      3 |               0.9768 |               530841 |                 0.9714 |                 530841 |
|      4 |               0.9795 |               484115 |                 0.976  |                 484115 |
|      5 |               0.9811 |               321957 |                 0.979  |                 321957 |
|      6 |               0.9819 |               164868 |                 0.9808 |                 164868 |
|      7 |               0.9826 |                63968 |                 0.9823 |                  63968 |
|      8 |               0.9827 |                23456 |                 0.9833 |                  23456 |

## Singletons vs non-singletons

| kind          |   chosen |   flat_0.5 |
|:--------------|---------:|-----------:|
| non-singleton |   0.975  |     0.9704 |
| singleton     |   0.9591 |     0.9074 |


exactness check: metrics.report 0.974106 vs fast 0.974106

saved decision config: {'t_keep': 0.7, 'ratio': 0.8, 't_null': 0.7, 'assign': True, 'cap': 11, 'cv_macro_f05': 0.9741057723544665, 'cv_by_country': {'India': 0.9672, 'US': 0.9787}}
