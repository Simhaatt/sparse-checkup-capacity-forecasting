# Round 4 report (Claude, cloud) — Luh-Blank, Sandia, Oxford, Kirkaldy

All choices were made on inner validation; every pre-declared arm is reported whatever its sign. Pre-declaration and amendments (with timestamps and the order in which results were seen) are in PLAN_R4.json; keys: created, data, R4A_origin_specific_search, R4B_refit, R4C_pooled, R4D_hybrid, combinations, statistics, R4E_transfer_hybrid_added, R4F_ablations_added, R4G_reviewer_robustness_added, R4A_compute_amendment, G7_compute_amendment.


**Compute amendment** (2026-10-07T13:46 UTC): Compute amendment (2-core container that restarts; ~13 CPU-hours left). R4A origin-specific search and finals restricted to origins 10 and 15, run in that order; origin 20 dropped because round 3 already performed the equal-budget retuning at origin 20 (same families, 20 settings each). R4C pooled arm restricted to Luh origins 10 and 15 and Sandia origins 5 and 8. Declared while the origin-10 search was at 289/500 trials; no R4A/R4C test-set result had been computed or seen. If compute remains after everything else finishes, origin 20 is run and reported as an addendum.


## 1. Luh-Blank: origin-specific re-search, refits and the hybrid (104 test cells)


### Equal-cell capacity MAPE by method and origin (round-3 references + round-4)

|   origin | method                 |   n_cells |   Q_MAPE |   ci95_low |   ci95_high |
|---------:|:-----------------------|----------:|---------:|-----------:|------------:|
|       10 | ARIMA_AICc             |       104 |   3.2809 |     2.6398 |      4.0203 |
|       10 | avg_ARIMA_XGB          |       104 |   3.2069 |     2.5802 |      3.9321 |
|       10 | avg_ARIMA_XGB_NN       |       104 |   2.742  |     2.2574 |      3.2698 |
|       10 | avg_ARIMA_hybrid       |       104 |   2.3483 |     1.8824 |      2.8691 |
|       10 | avg_ARIMA_r4NN         |       104 |   2.4    |     1.9486 |      2.8921 |
|       10 | avg_ARIMA_r4xgb        |       104 |   2.4391 |     1.9219 |      3.0156 |
|       10 | best_neural            |       104 |   3.7484 |     3.009  |      4.5219 |
|       10 | hybrid_sqrt_xgb        |       104 |   2.2242 |     1.7123 |      2.7684 |
|       10 | persistence            |       104 |   6.943  |     6.0686 |      7.8493 |
|       10 | r4_best_neural         |       104 |   2.6266 |     2.0968 |      3.2127 |
|       10 | r4_compact_lstm        |       104 |   2.7265 |     2.1824 |      3.3187 |
|       10 | r4_gru                 |       104 |   2.6908 |     2.1709 |      3.2458 |
|       10 | r4_large_lstm          |       104 |   2.6348 |     2.1598 |      3.132  |
|       10 | r4_rf                  |       104 |   3.8312 |     2.9363 |      4.7673 |
|       10 | r4_xgb                 |       104 |   2.8026 |     2.0881 |      3.5843 |
|       10 | regime_origin_selector |       104 |   2.4455 |     1.9569 |      2.969  |
|       10 | rf                     |       104 |   4.4012 |     3.3227 |      5.5558 |
|       10 | sqrt_k_lag1            |       104 |   3.193  |     2.6219 |      3.7694 |
|       10 | xgb                    |       104 |   4.6525 |     3.6884 |      5.6928 |
|       15 | ARIMA_AICc             |       104 |   2.7128 |     2.0855 |      3.3695 |
|       15 | avg_ARIMA_XGB          |       104 |   2.5009 |     1.9677 |      3.0607 |
|       15 | avg_ARIMA_XGB_NN       |       104 |   2.3559 |     1.8473 |      2.914  |
|       15 | avg_ARIMA_hybrid       |       104 |   2.1433 |     1.6608 |      2.6497 |
|       15 | avg_ARIMA_r4NN         |       104 |   2.1976 |     1.692  |      2.7459 |
|       15 | avg_ARIMA_r4xgb        |       104 |   2.4509 |     1.8943 |      3.0435 |
|       15 | best_neural            |       104 |   3.1101 |     2.4171 |      3.9006 |
|       15 | hybrid_sqrt_xgb        |       104 |   2.1851 |     1.7092 |      2.6822 |
|       15 | persistence            |       104 |   5.4839 |     4.5916 |      6.4246 |
|       15 | r4_best_neural         |       104 |   2.2325 |     1.7523 |      2.7578 |
|       15 | r4_compact_lstm        |       104 |   2.7637 |     2.0867 |      3.4921 |
|       15 | r4_gru                 |       104 |   2.4274 |     1.8735 |      3.0241 |
|       15 | r4_large_lstm          |       104 |   2.1587 |     1.7053 |      2.6567 |
|       15 | r4_rf                  |       104 |   2.4375 |     1.7833 |      3.1233 |
|       15 | r4_xgb                 |       104 |   2.7682 |     2.0965 |      3.4786 |
|       15 | regime_origin_selector |       104 |   2.4202 |     1.8896 |      2.999  |
|       15 | rf                     |       104 |   3.2582 |     2.4872 |      4.0731 |
|       15 | sqrt_k_lag1            |       104 |   3.1507 |     2.475  |      3.8536 |
|       15 | xgb                    |       104 |   3.217  |     2.5115 |      3.9841 |
|       20 | ARIMA_AICc             |       104 |   2.761  |     1.9569 |      3.6787 |
|       20 | avg_ARIMA_XGB          |       104 |   2.2866 |     1.7044 |      2.9341 |
|       20 | avg_ARIMA_XGB_NN       |       104 |   2.3486 |     1.7404 |      3.0348 |
|       20 | avg_ARIMA_hybrid       |       104 |   2.069  |     1.4887 |      2.7189 |
|       20 | best_neural            |       104 |   3.6311 |     2.6989 |      4.6673 |
|       20 | gru__history_only      |       104 |   2.489  |     1.7945 |      3.2647 |
|       20 | hybrid_sqrt_xgb        |       104 |   2.1821 |     1.6027 |      2.8264 |
|       20 | persistence            |       104 |   4.8771 |     3.7767 |      6.0873 |
|       20 | regime_origin_selector |       104 |   2.9986 |     2.1864 |      3.9082 |
|       20 | rf                     |       104 |   2.7259 |     1.9628 |      3.5845 |
|       20 | sqrt_k_lag1            |       104 |   4.4899 |     3.2498 |      5.8873 |
|       20 | xgb                    |       104 |   2.7228 |     2.0741 |      3.4675 |

### Paired tests vs ARIMA and vs round-3 best combination (Holm within family)

| family                    |   origin | comparison                       |   n |   diff_pp |   ci95_low |   ci95_high |      p |   p_holm |
|:--------------------------|---------:|:---------------------------------|----:|----------:|-----------:|------------:|-------:|---------:|
| r4_vs_ARIMA_origin10      |       10 | avg_ARIMA_hybrid - ARIMA_AICc    | 104 |   -0.9326 |    -1.2498 |     -0.639  | 0      |   0      |
| r4_vs_ARIMA_origin10      |       10 | avg_ARIMA_r4NN - ARIMA_AICc      | 104 |   -0.8809 |    -1.2891 |     -0.5118 | 0      |   0      |
| r4_vs_ARIMA_origin10      |       10 | avg_ARIMA_r4xgb - ARIMA_AICc     | 104 |   -0.8418 |    -1.2436 |     -0.4773 | 0      |   0      |
| r4_vs_ARIMA_origin10      |       10 | hybrid_sqrt_xgb - ARIMA_AICc     | 104 |   -1.0567 |    -1.6808 |     -0.4724 | 0.0002 |   0.0016 |
| r4_vs_ARIMA_origin10      |       10 | r4_best_neural - ARIMA_AICc      | 104 |   -0.6544 |    -1.3696 |     -0.0001 | 0.0334 |   0.1202 |
| r4_vs_ARIMA_origin10      |       10 | r4_compact_lstm - ARIMA_AICc     | 104 |   -0.5545 |    -1.3127 |      0.1308 | 0.0618 |   0.1236 |
| r4_vs_ARIMA_origin10      |       10 | r4_gru - ARIMA_AICc              | 104 |   -0.5902 |    -1.283  |      0.0489 | 0.0234 |   0.117  |
| r4_vs_ARIMA_origin10      |       10 | r4_large_lstm - ARIMA_AICc       | 104 |   -0.6461 |    -1.3387 |     -0.0509 | 0.03   |   0.1202 |
| r4_vs_ARIMA_origin10      |       10 | r4_rf - ARIMA_AICc               | 104 |    0.5502 |    -0.1528 |      1.2886 | 0.7359 |   0.7359 |
| r4_vs_ARIMA_origin10      |       10 | r4_xgb - ARIMA_AICc              | 104 |   -0.4784 |    -1.2145 |      0.2029 | 0.0047 |   0.0284 |
| r4_vs_round3best_origin10 |       10 | avg_ARIMA_hybrid - avg_ARIMA_XGB | 104 |   -0.8586 |    -1.2515 |     -0.4969 | 0.0001 |   0.0006 |
| r4_vs_round3best_origin10 |       10 | avg_ARIMA_r4NN - avg_ARIMA_XGB   | 104 |   -0.8068 |    -1.2991 |     -0.3481 | 0.0084 |   0.0587 |
| r4_vs_round3best_origin10 |       10 | avg_ARIMA_r4xgb - avg_ARIMA_XGB  | 104 |   -0.7678 |    -1.1893 |     -0.3775 | 0      |   0.0002 |
| r4_vs_round3best_origin10 |       10 | hybrid_sqrt_xgb - avg_ARIMA_XGB  | 104 |   -0.9827 |    -1.5239 |     -0.4918 | 0      |   0.0001 |
| r4_vs_round3best_origin10 |       10 | r4_best_neural - avg_ARIMA_XGB   | 104 |   -0.5803 |    -1.2937 |      0.0626 | 0.0561 |   0.2807 |
| r4_vs_round3best_origin10 |       10 | r4_compact_lstm - avg_ARIMA_XGB  | 104 |   -0.4804 |    -1.2094 |      0.1357 | 0.0956 |   0.2959 |
| r4_vs_round3best_origin10 |       10 | r4_gru - avg_ARIMA_XGB           | 104 |   -0.5161 |    -1.2209 |      0.1171 | 0.074  |   0.2959 |
| r4_vs_round3best_origin10 |       10 | r4_large_lstm - avg_ARIMA_XGB    | 104 |   -0.5721 |    -1.2586 |      0.0064 | 0.1099 |   0.2959 |
| r4_vs_round3best_origin10 |       10 | r4_rf - avg_ARIMA_XGB            | 104 |    0.6243 |    -0.0009 |      1.2817 | 0.3937 |   0.3937 |
| r4_vs_round3best_origin10 |       10 | r4_xgb - avg_ARIMA_XGB           | 104 |   -0.4043 |    -1.1187 |      0.2514 | 0.0106 |   0.0637 |
| r4_vs_ARIMA_origin15      |       15 | avg_ARIMA_hybrid - ARIMA_AICc    | 104 |   -0.5695 |    -0.8213 |     -0.3373 | 0      |   0      |
| r4_vs_ARIMA_origin15      |       15 | avg_ARIMA_r4NN - ARIMA_AICc      | 104 |   -0.5152 |    -0.7757 |     -0.2747 | 0.0005 |   0.0043 |
| r4_vs_ARIMA_origin15      |       15 | avg_ARIMA_r4xgb - ARIMA_AICc     | 104 |   -0.2619 |    -0.5341 |      0.0031 | 0.0936 |   0.6554 |
| r4_vs_ARIMA_origin15      |       15 | hybrid_sqrt_xgb - ARIMA_AICc     | 104 |   -0.5277 |    -0.9874 |     -0.097  | 0.0359 |   0.2872 |
| r4_vs_ARIMA_origin15      |       15 | r4_best_neural - ARIMA_AICc      | 104 |   -0.4803 |    -0.9432 |     -0.055  | 0.4231 |   1      |
| r4_vs_ARIMA_origin15      |       15 | r4_compact_lstm - ARIMA_AICc     | 104 |    0.0509 |    -0.4399 |      0.5426 | 0.3437 |   1      |
| r4_vs_ARIMA_origin15      |       15 | r4_gru - ARIMA_AICc              | 104 |   -0.2854 |    -0.7579 |      0.1644 | 0.446  |   1      |
| r4_vs_ARIMA_origin15      |       15 | r4_large_lstm - ARIMA_AICc       | 104 |   -0.5542 |    -1.0174 |     -0.1332 | 0.1603 |   0.8014 |
| r4_vs_ARIMA_origin15      |       15 | r4_rf - ARIMA_AICc               | 104 |   -0.2753 |    -0.8119 |      0.2722 | 0.1219 |   0.7314 |
| r4_vs_ARIMA_origin15      |       15 | r4_xgb - ARIMA_AICc              | 104 |    0.0554 |    -0.5288 |      0.643  | 0.6569 |   1      |
| r4_vs_round3best_origin15 |       15 | avg_ARIMA_hybrid - avg_ARIMA_XGB | 104 |   -0.3576 |    -0.6475 |     -0.0667 | 0.0111 |   0.1111 |
| r4_vs_round3best_origin15 |       15 | avg_ARIMA_r4NN - avg_ARIMA_XGB   | 104 |   -0.3033 |    -0.631  |      0.0277 | 0.0329 |   0.2958 |
| r4_vs_round3best_origin15 |       15 | avg_ARIMA_r4xgb - avg_ARIMA_XGB  | 104 |   -0.05   |    -0.3293 |      0.248  | 0.1935 |   0.9674 |
| r4_vs_round3best_origin15 |       15 | hybrid_sqrt_xgb - avg_ARIMA_XGB  | 104 |   -0.3158 |    -0.6889 |      0.0467 | 0.0596 |   0.4764 |
| r4_vs_round3best_origin15 |       15 | r4_best_neural - avg_ARIMA_XGB   | 104 |   -0.2684 |    -0.6939 |      0.1452 | 0.224  |   0.9674 |
| r4_vs_round3best_origin15 |       15 | r4_compact_lstm - avg_ARIMA_XGB  | 104 |    0.2629 |    -0.1818 |      0.7284 | 0.6129 |   1      |
| r4_vs_round3best_origin15 |       15 | r4_gru - avg_ARIMA_XGB           | 104 |   -0.0735 |    -0.512  |      0.3508 | 0.828  |   1      |
| r4_vs_round3best_origin15 |       15 | r4_large_lstm - avg_ARIMA_XGB    | 104 |   -0.3422 |    -0.7396 |      0.046  | 0.1056 |   0.6338 |
| r4_vs_round3best_origin15 |       15 | r4_rf - avg_ARIMA_XGB            | 104 |   -0.0634 |    -0.5598 |      0.4465 | 0.0839 |   0.5874 |
| r4_vs_round3best_origin15 |       15 | r4_xgb - avg_ARIMA_XGB           | 104 |    0.2673 |    -0.2066 |      0.7371 | 0.307  |   0.9674 |
| r4_vs_ARIMA_origin20      |       20 | avg_ARIMA_hybrid - ARIMA_AICc    | 104 |   -0.692  |    -1.0957 |     -0.3507 | 0.0003 |   0.0005 |
| r4_vs_ARIMA_origin20      |       20 | hybrid_sqrt_xgb - ARIMA_AICc     | 104 |   -0.5789 |    -1.205  |     -0.0172 | 0.1763 |   0.1763 |
| r4_vs_round3best_origin20 |       20 | avg_ARIMA_hybrid - avg_ARIMA_XGB | 104 |   -0.2177 |    -0.4602 |      0.0197 | 0.0068 |   0.0137 |
| r4_vs_round3best_origin20 |       20 | hybrid_sqrt_xgb - avg_ARIMA_XGB  | 104 |   -0.1046 |    -0.4572 |      0.2643 | 0.0772 |   0.0772 |

### Post hoc: hybrid vs origin-specifically re-tuned learned models (Holm within these 8)

|   origin | comparison                        |   n |   diff_pp |      p |   p_holm |
|---------:|:----------------------------------|----:|----------:|-------:|---------:|
|       10 | hybrid_sqrt_xgb - r4_best_neural  | 104 |   -0.4024 | 0.0009 |   0.0061 |
|       10 | hybrid_sqrt_xgb - r4_large_lstm   | 104 |   -0.4106 | 0.0007 |   0.0055 |
|       10 | hybrid_sqrt_xgb - r4_xgb          | 104 |   -0.5784 | 0.0221 |   0.1323 |
|       10 | avg_ARIMA_hybrid - avg_ARIMA_r4NN | 104 |   -0.0517 | 0.2289 |   0.9158 |
|       15 | hybrid_sqrt_xgb - r4_best_neural  | 104 |   -0.0474 | 0.4213 |   1      |
|       15 | hybrid_sqrt_xgb - r4_large_lstm   | 104 |    0.0265 | 0.6244 |   1      |
|       15 | hybrid_sqrt_xgb - r4_xgb          | 104 |   -0.5831 | 0.0655 |   0.3274 |
|       15 | avg_ARIMA_hybrid - avg_ARIMA_r4NN | 104 |   -0.0543 | 0.8179 |   1      |

### Validation-chosen ARIMA weight w*ARIMA+(1-w)*hybrid (G6)

| dataset   |   origin |   weighted |   hybrid |   ARIMA |   diff_vs_ARIMA |   p_vs_ARIMA |   diff_vs_hybrid |   p_vs_hybrid |   p_vs_ARIMA_holm |
|:----------|---------:|-----------:|---------:|--------:|----------------:|-------------:|-----------------:|--------------:|------------------:|
| luh       |       10 |     2.1621 |   2.2242 |  3.2809 |         -1.1189 |       0      |          -0.0621 |        0.5418 |            0      |
| luh       |       15 |     2.0758 |   2.1851 |  2.7128 |         -0.637  |       0.0003 |          -0.1094 |        0.6525 |            0.0005 |
| luh       |       20 |     2.1125 |   2.1821 |  2.761  |         -0.6485 |       0.0025 |          -0.0696 |        0.827  |            0.0025 |
| snl       |        5 |     1.5037 |   1.5037 |  4.5683 |         -3.0647 |       0      |           0      |        1      |            0      |
| snl       |        8 |     1.8938 |   1.7035 |  3.1743 |         -1.2804 |       0      |           0.1903 |        0.7499 |            0      |

### G6 chosen weights

| dataset   |   origin |   fold |    w |
|:----------|---------:|-------:|-----:|
| luh       |       10 |      1 | 0.25 |
| luh       |       10 |      2 | 0.75 |
| luh       |       10 |      3 | 0    |
| luh       |       10 |      4 | 0    |
| luh       |       10 |      5 | 0.25 |
| luh       |       15 |      1 | 0.5  |
| luh       |       15 |      2 | 0.75 |
| luh       |       15 |      3 | 0    |
| luh       |       15 |      4 | 0    |
| luh       |       15 |      5 | 0.5  |
| luh       |       20 |      1 | 0.25 |
| luh       |       20 |      2 | 0.75 |
| luh       |       20 |      3 | 0    |
| luh       |       20 |      4 | 0.25 |
| luh       |       20 |      5 | 0.5  |
| snl       |        5 |      1 | 0    |
| snl       |        5 |      2 | 0    |
| snl       |        5 |      3 | 0    |
| snl       |        5 |      4 | 0    |
| snl       |        5 |      5 | 0    |
| snl       |        8 |      1 | 0.5  |
| snl       |        8 |      2 | 0    |
| snl       |        8 |      3 | 0.25 |
| snl       |        8 |      4 | 0.25 |
| snl       |        8 |      5 | 0.5  |

## 2. Why it works: ablations (R4F)


### Luh ablation summary

| variant               |     10 |     15 |     20 |
|:----------------------|-------:|-------:|-------:|
| V1_main               | 2.293  | 2.2183 | 2.2229 |
| V2_no_diag            | 2.3893 | 2.3638 | 2.5279 |
| V3_single_origin      | 2.5193 | 2.1787 | 2.2856 |
| V4_linear_prior       | 2.3466 | 2.3319 | 2.1767 |
| V5_persistence_anchor | 2.4917 | 2.3792 | 2.0946 |
| V6_xuan_style         | 2.7053 | 2.5885 | 2.489  |

### Luh ablations vs V1 (Holm within origin)

| dataset   |   origin | comparison                      |   n |   diff_pp |      p |   p_holm |
|:----------|---------:|:--------------------------------|----:|----------:|-------:|---------:|
| luh       |       10 | V2_no_diag - V1_main            | 104 |    0.0964 | 0.0451 |   0.1352 |
| luh       |       10 | V3_single_origin - V1_main      | 104 |    0.2263 | 0.2646 |   0.5293 |
| luh       |       10 | V4_linear_prior - V1_main       | 104 |    0.0536 | 0.2802 |   0.5293 |
| luh       |       10 | V5_persistence_anchor - V1_main | 104 |    0.1987 | 0.0219 |   0.0875 |
| luh       |       10 | V6_xuan_style - V1_main         | 104 |    0.4124 | 0.0026 |   0.0132 |
| luh       |       15 | V2_no_diag - V1_main            | 104 |    0.1456 | 0.0052 |   0.0259 |
| luh       |       15 | V3_single_origin - V1_main      | 104 |   -0.0395 | 0.2142 |   0.4143 |
| luh       |       15 | V4_linear_prior - V1_main       | 104 |    0.1136 | 0.0092 |   0.0369 |
| luh       |       15 | V5_persistence_anchor - V1_main | 104 |    0.1609 | 0.2071 |   0.4143 |
| luh       |       15 | V6_xuan_style - V1_main         | 104 |    0.3702 | 0.0286 |   0.0858 |
| luh       |       20 | V2_no_diag - V1_main            | 104 |    0.305  | 0.0006 |   0.0029 |
| luh       |       20 | V3_single_origin - V1_main      | 104 |    0.0626 | 0.0098 |   0.0394 |
| luh       |       20 | V4_linear_prior - V1_main       | 104 |   -0.0462 | 0.255  |   0.5101 |
| luh       |       20 | V5_persistence_anchor - V1_main | 104 |   -0.1283 | 0.3919 |   0.5101 |
| luh       |       20 | V6_xuan_style - V1_main         | 104 |    0.266  | 0.0613 |   0.184  |

### Sandia ablation summary

| variant               |      5 |      8 |
|:----------------------|-------:|-------:|
| V1_main               | 1.5544 | 1.3859 |
| V2_no_diag            | 1.4668 | 1.4325 |
| V3_single_origin      | 1.4722 | 1.8047 |
| V4_linear_prior       | 1.6419 | 1.5179 |
| V5_persistence_anchor | 1.439  | 1.7782 |
| V6_xuan_style         | 1.9221 | 1.9282 |

### Sandia ablations vs V1

| dataset   |   origin | comparison                      |   n |   diff_pp |      p |   p_holm |
|:----------|---------:|:--------------------------------|----:|----------:|-------:|---------:|
| snl       |        5 | V2_no_diag - V1_main            |  61 |   -0.0876 | 0.1735 |   0.403  |
| snl       |        5 | V3_single_origin - V1_main      |  61 |   -0.0822 | 0.1068 |   0.403  |
| snl       |        5 | V4_linear_prior - V1_main       |  61 |    0.0875 | 0.1827 |   0.403  |
| snl       |        5 | V5_persistence_anchor - V1_main |  61 |   -0.1154 | 0.1007 |   0.403  |
| snl       |        5 | V6_xuan_style - V1_main         |  61 |    0.3677 | 0.0432 |   0.2159 |
| snl       |        8 | V2_no_diag - V1_main            |  39 |    0.0466 | 0.3222 |   0.3825 |
| snl       |        8 | V3_single_origin - V1_main      |  39 |    0.4188 | 0.1275 |   0.3825 |
| snl       |        8 | V4_linear_prior - V1_main       |  39 |    0.132  | 0.1423 |   0.3825 |
| snl       |        8 | V5_persistence_anchor - V1_main |  39 |    0.3923 | 0.0901 |   0.3606 |
| snl       |        8 | V6_xuan_style - V1_main         |  39 |    0.5423 | 0.0111 |   0.0555 |

### Hybrid without the test schedule (checkup index only, G3)

| dataset   |   origin |   MAPE |
|:----------|---------:|-------:|
| luh       |       10 | 1.9257 |
| luh       |       15 | 1.9768 |
| luh       |       20 | 2.016  |
| snl       |        5 | 1.6565 |
| snl       |        8 | 1.8692 |

### G3 minus main hybrid (paired, per origin)

| dataset   |   origin |   n |   diff_noschedule_minus_main |      p |
|:----------|---------:|----:|-----------------------------:|-------:|
| luh       |       10 | 104 |                      -0.2985 | 0.2934 |
| luh       |       15 | 104 |                      -0.2084 | 0.0098 |
| luh       |       20 | 104 |                      -0.166  | 0.0887 |
| snl       |        5 |  61 |                       0.1529 | 0.849  |
| snl       |        8 |  39 |                       0.1657 | 0.3872 |

## 3. Sandia (80 cells)


### Sandia hybrid paired tests

|   origin | comparison                               |   n |    diff |      p |   p_holm |
|---------:|:-----------------------------------------|----:|--------:|-------:|---------:|
|        5 | hybrid_sqrt_xgb - ARIMA_AICc             |  61 | -3.0647 | 0      |   0      |
|        5 | hybrid_sqrt_xgb - sqrt_efc_lag1          |  61 | -0.9577 | 0      |   0      |
|        5 | hybrid_sqrt_xgb - C_sqrt_xgb_nn          |  61 | -0.097  | 0.8434 |   1      |
|        5 | hybrid_sqrt_xgb - C_all_learned          |  61 | -0.1056 | 0.7711 |   1      |
|        5 | hybrid_sqrt_xgb - selector_by_chemistry  |  61 | -0.3572 | 0.2927 |   1      |
|        5 | hybrid_sqrt_xgb - rf                     |  61 | -0.3561 | 0.356  |   1      |
|        5 | avg_ARIMA_hybrid - ARIMA_AICc            |  61 | -2.1227 | 0      |   0      |
|        5 | avg_ARIMA_hybrid - sqrt_efc_lag1         |  61 | -0.0157 | 0.6796 |   1      |
|        5 | avg_ARIMA_hybrid - C_sqrt_xgb_nn         |  61 |  0.845  | 0      |   0      |
|        5 | avg_ARIMA_hybrid - C_all_learned         |  61 |  0.8364 | 0      |   0.0001 |
|        5 | avg_ARIMA_hybrid - selector_by_chemistry |  61 |  0.5847 | 0.0029 |   0.0175 |
|        5 | avg_ARIMA_hybrid - rf                    |  61 |  0.5859 | 0.0001 |   0.0007 |
|        8 | hybrid_sqrt_xgb - ARIMA_AICc             |  39 | -1.4707 | 0.0002 |   0.0017 |
|        8 | hybrid_sqrt_xgb - sqrt_efc_lag1          |  39 | -0.8917 | 0.0047 |   0.0472 |
|        8 | hybrid_sqrt_xgb - C_sqrt_xgb_nn          |  39 | -0.1717 | 0.0093 |   0.0748 |
|        8 | hybrid_sqrt_xgb - C_all_learned          |  39 | -0.1373 | 0.0062 |   0.0562 |
|        8 | hybrid_sqrt_xgb - selector_by_chemistry  |  39 |  0.0349 | 0.3949 |   1      |
|        8 | hybrid_sqrt_xgb - rf                     |  39 | -0.5377 | 0.0143 |   0.1    |
|        8 | avg_ARIMA_hybrid - ARIMA_AICc            |  39 | -1.4245 | 0      |   0      |
|        8 | avg_ARIMA_hybrid - sqrt_efc_lag1         |  39 | -0.8455 | 0.019  |   0.1137 |
|        8 | avg_ARIMA_hybrid - C_sqrt_xgb_nn         |  39 | -0.1255 | 0.2706 |   1      |
|        8 | avg_ARIMA_hybrid - C_all_learned         |  39 | -0.0911 | 0.3647 |   1      |
|        8 | avg_ARIMA_hybrid - selector_by_chemistry |  39 |  0.0811 | 0.8092 |   1      |
|        8 | avg_ARIMA_hybrid - rf                    |  39 | -0.4915 | 0.2357 |   1      |

## 4. New cell formats without target data


### Kirkaldy zero-target transfer

|   origin | method          |   n_cells |    MAPE |
|---------:|:----------------|----------:|--------:|
|        3 | ARIMA_AICc      |        40 | 10.0527 |
|        3 | hybrid_transfer |        40 |  4.0816 |
|        3 | persistence     |        40 | 10.0527 |
|        3 | sqrt_idx_prior  |        40 |  4.475  |
|        5 | ARIMA_AICc      |        40 |  7.479  |
|        5 | hybrid_transfer |        40 |  3.6053 |
|        5 | persistence     |        40 |  7.783  |
|        5 | sqrt_idx_prior  |        40 |  4.3643 |
|        8 | ARIMA_AICc      |        40 |  3.788  |
|        8 | hybrid_transfer |        40 |  3.3887 |
|        8 | persistence     |        40 |  6.278  |
|        8 | sqrt_idx_prior  |        40 |  4.7153 |

### Kirkaldy paired tests

|   origin | comparison                       |   n |   diff_pp |      p |   p_holm |
|---------:|:---------------------------------|----:|----------:|-------:|---------:|
|        3 | hybrid_transfer - sqrt_idx_prior |  40 |   -0.3934 | 0.2064 |   0.2064 |
|        3 | hybrid_transfer - ARIMA_AICc     |  40 |   -5.9711 | 0      |   0      |
|        3 | hybrid_transfer - persistence    |  40 |   -5.9711 | 0      |   0      |
|        5 | hybrid_transfer - sqrt_idx_prior |  40 |   -0.759  | 0.0111 |   0.0111 |
|        5 | hybrid_transfer - ARIMA_AICc     |  40 |   -3.8737 | 0      |   0      |
|        5 | hybrid_transfer - persistence    |  40 |   -4.1777 | 0      |   0      |
|        8 | hybrid_transfer - sqrt_idx_prior |  40 |   -1.3265 | 0.0259 |   0.0518 |
|        8 | hybrid_transfer - ARIMA_AICc     |  40 |   -0.3993 | 0.2649 |   0.2649 |
|        8 | hybrid_transfer - persistence    |  40 |   -2.8892 | 0      |   0      |

### Kirkaldy transfer by source set

| method                     |       3 |      5 |      8 |
|:---------------------------|--------:|-------:|-------:|
| ARIMA_AICc                 | 10.0527 | 7.479  | 3.788  |
| anchor_persistence_src_all |  4.8777 | 3.8282 | 3.2186 |
| hybrid_src_LB              |  4.1719 | 3.7555 | 3.1656 |
| hybrid_src_LB+SNL          |  4.1285 | 3.7218 | 3.2307 |
| hybrid_src_LB+SNL+OX       |  4.0816 | 3.6053 | 3.3887 |
| hybrid_src_SNL             |  3.8846 | 3.8656 | 4.0361 |
| persistence                | 10.0527 | 7.783  | 6.278  |
| sqrt_idx_prior             |  4.475  | 4.3643 | 4.7153 |

### Kirkaldy transfer source ablation

|   origin | comparison                                  |   diff_pp |      p |   p_holm |
|---------:|:--------------------------------------------|----------:|-------:|---------:|
|        3 | anchor_persistence_src_all - sqrt_idx_prior |    0.4026 | 0.5099 |   1      |
|        3 | hybrid_src_LB - sqrt_idx_prior              |   -0.3032 | 0.5014 |   1      |
|        3 | hybrid_src_LB+SNL - sqrt_idx_prior          |   -0.3466 | 0.3011 |   1      |
|        3 | hybrid_src_LB+SNL+OX - sqrt_idx_prior       |   -0.3934 | 0.2064 |   1      |
|        3 | hybrid_src_SNL - sqrt_idx_prior             |   -0.5904 | 0.6463 |   1      |
|        5 | anchor_persistence_src_all - sqrt_idx_prior |   -0.5361 | 0.1659 |   0.3318 |
|        5 | hybrid_src_LB - sqrt_idx_prior              |   -0.6088 | 0.009  |   0.045  |
|        5 | hybrid_src_LB+SNL - sqrt_idx_prior          |   -0.6425 | 0.0299 |   0.0897 |
|        5 | hybrid_src_LB+SNL+OX - sqrt_idx_prior       |   -0.759  | 0.0111 |   0.045  |
|        5 | hybrid_src_SNL - sqrt_idx_prior             |   -0.4987 | 0.2707 |   0.3318 |
|        8 | anchor_persistence_src_all - sqrt_idx_prior |   -1.4966 | 0.0006 |   0.003  |
|        8 | hybrid_src_LB - sqrt_idx_prior              |   -1.5497 | 0.0584 |   0.0789 |
|        8 | hybrid_src_LB+SNL - sqrt_idx_prior          |   -1.4846 | 0.0394 |   0.0789 |
|        8 | hybrid_src_LB+SNL+OX - sqrt_idx_prior       |   -1.3265 | 0.0259 |   0.0777 |
|        8 | hybrid_src_SNL - sqrt_idx_prior             |   -0.6792 | 0.0086 |   0.0345 |

### Oxford frozen-design confirmation (no Oxford data used, G1)

| method         |       5 |      10 |     20 |
|:---------------|--------:|--------:|-------:|
| ARIMA_AICc     | 13.7851 |  5.6734 | 4.841  |
| hybrid_frozen  |  4.3961 |  5.0651 | 1.8671 |
| persistence    | 13.3079 | 12.1761 | 9.038  |
| sqrt_idx_prior |  7.8077 |  7.3219 | 5.3782 |

### Oxford error by horizon bucket

|   origin | hbin   |   ARIMA_AICc |   hybrid_frozen |   persistence |   sqrt_idx_prior |
|---------:|:-------|-------------:|----------------:|--------------:|-----------------:|
|        5 | 1-5    |       1.2798 |          0.3189 |        1.2398 |           0.4426 |
|        5 | 6-15   |       4.8121 |          0.9666 |        4.697  |           2.2102 |
|        5 | 16-30  |      10.749  |          2.9672 |       10.57   |           5.9156 |
|        5 | 31+    |      21.381  |          7.1289 |       20.3777 |          12.2736 |
|       10 | 1-5    |       0.4249 |          0.5727 |        1.1951 |           0.6877 |
|       10 | 6-15   |       0.981  |          2.1853 |        5.1503 |           3.0582 |
|       10 | 16-30  |       3.69   |          4.2556 |       10.4642 |           6.1398 |
|       10 | 31+    |      10.9556 |          7.8316 |       19.2191 |          11.6225 |
|       20 | 1-5    |       0.33   |          0.3627 |        1.267  |           1.6639 |
|       20 | 6-15   |       1.6534 |          0.6304 |        4.2646 |           2.9314 |
|       20 | 16-30  |       4.2203 |          1.6247 |        9.4857 |           5.669  |
|       20 | 31+    |       9.8012 |          3.5645 |       15.8178 |           8.6185 |

### Oxford paired tests (n=8; minimum attainable p = 0.0078)

|   origin | comparison                     |   n |    diff |   wins |      p |
|---------:|:-------------------------------|----:|--------:|-------:|-------:|
|        5 | hybrid_frozen - ARIMA_AICc     |   8 | -9.389  |      8 | 0.0078 |
|        5 | hybrid_frozen - sqrt_idx_prior |   8 | -3.4116 |      8 | 0.0078 |
|        5 | hybrid_frozen - persistence    |   8 | -8.9118 |      8 | 0.0078 |
|       10 | hybrid_frozen - ARIMA_AICc     |   8 | -0.6083 |      3 | 0.8438 |
|       10 | hybrid_frozen - sqrt_idx_prior |   8 | -2.2568 |      8 | 0.0078 |
|       10 | hybrid_frozen - persistence    |   8 | -7.111  |      8 | 0.0078 |
|       20 | hybrid_frozen - ARIMA_AICc     |   8 | -2.9739 |      8 | 0.0078 |
|       20 | hybrid_frozen - sqrt_idx_prior |   8 | -3.5111 |      8 | 0.0078 |
|       20 | hybrid_frozen - persistence    |   8 | -7.1709 |      8 | 0.0078 |

## 5. Pooling public datasets (R4C)


### Pooled vs source-only history-only models

| dataset   | method          |        5 |        8 |       10 |       15 |
|:----------|:----------------|---------:|---------:|---------:|---------:|
| luh       | ARIMA_AICc      | nan      | nan      |   3.2809 |   2.7128 |
| luh       | gru_pooled      | nan      | nan      |   3.2474 |   2.706  |
| luh       | gru_source_only | nan      | nan      |   3.0053 |   2.7317 |
| luh       | hybrid_sqrt_xgb | nan      | nan      |   2.2242 |   2.1851 |
| luh       | xgb_pooled      | nan      | nan      |   3.7825 |   3.2708 |
| luh       | xgb_source_only | nan      | nan      |   3.5775 |   3.1793 |
| snl       | ARIMA_AICc      |   4.5683 |   3.1743 | nan      | nan      |
| snl       | gru_pooled      |   2.7051 |   3.1898 | nan      | nan      |
| snl       | gru_source_only |   2.2893 |   2.5055 | nan      | nan      |
| snl       | hybrid_sqrt_xgb |   1.5037 |   1.7035 | nan      | nan      |
| snl       | xgb_pooled      |   3.0202 |   3.6048 | nan      | nan      |
| snl       | xgb_source_only |   3.3893 |   3.776  | nan      | nan      |

### Pooled vs source-only paired tests

| dataset   |   origin | comparison                        |   n |   diff_pp |      p |   p_holm |
|:----------|---------:|:----------------------------------|----:|----------:|-------:|---------:|
| luh       |       10 | gru_pooled - gru_source_only      | 104 |    0.2421 | 0.1612 |   0.1612 |
| luh       |       10 | gru_pooled - hybrid_sqrt_xgb      | 104 |    1.0232 | 0      |   0      |
| luh       |       10 | gru_source_only - hybrid_sqrt_xgb | 104 |    0.7811 | 0.0001 |   0.0002 |
| luh       |       10 | xgb_pooled - xgb_source_only      | 104 |    0.205  | 0.0049 |   0.0099 |
| luh       |       10 | xgb_pooled - hybrid_sqrt_xgb      | 104 |    1.5583 | 0      |   0      |
| luh       |       10 | xgb_source_only - hybrid_sqrt_xgb | 104 |    1.3533 | 0      |   0      |
| luh       |       15 | gru_pooled - gru_source_only      | 104 |   -0.0257 | 0.7165 |   0.7165 |
| luh       |       15 | gru_pooled - hybrid_sqrt_xgb      | 104 |    0.5209 | 0.0404 |   0.1213 |
| luh       |       15 | gru_source_only - hybrid_sqrt_xgb | 104 |    0.5466 | 0.0182 |   0.0729 |
| luh       |       15 | xgb_pooled - xgb_source_only      | 104 |    0.0915 | 0.0618 |   0.1236 |
| luh       |       15 | xgb_pooled - hybrid_sqrt_xgb      | 104 |    1.0857 | 0.0008 |   0.0049 |
| luh       |       15 | xgb_source_only - hybrid_sqrt_xgb | 104 |    0.9942 | 0.0087 |   0.0435 |
| snl       |        5 | gru_pooled - gru_source_only      |  61 |    0.4158 | 0.0731 |   0.1462 |
| snl       |        5 | gru_pooled - hybrid_sqrt_xgb      |  61 |    1.2015 | 0      |   0.0001 |
| snl       |        5 | gru_source_only - hybrid_sqrt_xgb |  61 |    0.7857 | 0.0025 |   0.0076 |
| snl       |        5 | xgb_pooled - xgb_source_only      |  61 |   -0.3692 | 0.1234 |   0.1462 |
| snl       |        5 | xgb_pooled - hybrid_sqrt_xgb      |  61 |    1.5165 | 0      |   0      |
| snl       |        5 | xgb_source_only - hybrid_sqrt_xgb |  61 |    1.8857 | 0      |   0      |
| snl       |        8 | gru_pooled - gru_source_only      |  39 |    0.6843 | 0.0875 |   0.0875 |
| snl       |        8 | gru_pooled - hybrid_sqrt_xgb      |  39 |    1.4863 | 0      |   0      |
| snl       |        8 | gru_source_only - hybrid_sqrt_xgb |  39 |    0.802  | 0.0007 |   0.0022 |
| snl       |        8 | xgb_pooled - xgb_source_only      |  39 |   -0.1713 | 0.0054 |   0.0109 |
| snl       |        8 | xgb_pooled - hybrid_sqrt_xgb      |  39 |    1.9012 | 0.0001 |   0.0003 |
| snl       |        8 | xgb_source_only - hybrid_sqrt_xgb |  39 |    2.0725 | 0      |   0      |

## 6. Foundation-model baseline fine-tuned on sparse checkups


### Fine-tuned Chronos-Bolt-small (decoder fine-tuned, encoder frozen; checkpoint by validation)

| method                                  |   origin |   MAPE | dataset   |   q10_q90_coverage |
|:----------------------------------------|---------:|-------:|:----------|-------------------:|
| ARIMA_AICc                              |        5 | 4.5683 | snl       |           nan      |
| ARIMA_AICc                              |        8 | 3.1743 | snl       |           nan      |
| ChronosBolt_small_finetuned             |        5 | 2.9137 | snl       |             0.7839 |
| ChronosBolt_small_finetuned             |        8 | 3.0402 | snl       |             0.691  |
| ChronosBolt_small_finetuned_full_lr1e-5 |        5 | 2.7698 | snl       |             0.8226 |
| ChronosBolt_small_finetuned_full_lr1e-5 |        8 | 3.086  | snl       |             0.6602 |
| ChronosBolt_small_zero                  |        5 | 4.8817 | snl       |           nan      |
| ChronosBolt_small_zero                  |        8 | 4.9604 | snl       |           nan      |
| hybrid_sqrt_xgb                         |        5 | 1.5037 | snl       |           nan      |
| hybrid_sqrt_xgb                         |        8 | 1.7035 | snl       |           nan      |
| ARIMA_AICc                              |       10 | 3.2809 | luh       |           nan      |
| ARIMA_AICc                              |       15 | 2.7128 | luh       |           nan      |
| ARIMA_AICc                              |       20 | 2.761  | luh       |           nan      |
| ChronosBolt_small_finetuned             |       10 | 2.5375 | luh       |             0.8225 |
| ChronosBolt_small_finetuned             |       15 | 2.4514 | luh       |             0.8118 |
| ChronosBolt_small_finetuned             |       20 | 2.4424 | luh       |             0.8711 |
| hybrid_sqrt_xgb                         |       10 | 2.2242 | luh       |           nan      |
| hybrid_sqrt_xgb                         |       15 | 2.1851 | luh       |           nan      |
| hybrid_sqrt_xgb                         |       20 | 2.1821 | luh       |           nan      |

### Fine-tuned Chronos paired tests

| dataset   |   origin | comparison                                                       |   n |   diff_pp |      p |   p_holm |
|:----------|---------:|:-----------------------------------------------------------------|----:|----------:|-------:|---------:|
| snl       |        5 | ChronosBolt_small_finetuned - hybrid_sqrt_xgb                    |  61 |    1.4101 | 0      |   0.0001 |
| snl       |        5 | ChronosBolt_small_finetuned - ARIMA_AICc                         |  61 |   -1.6546 | 0      |   0      |
| snl       |        5 | ChronosBolt_small_finetuned - ChronosBolt_small_zero             |  61 |   -1.968  | 0      |   0      |
| snl       |        5 | ChronosBolt_small_finetuned_full_lr1e-5 - hybrid_sqrt_xgb        |  61 |    1.2661 | 0.0003 |   0.0003 |
| snl       |        5 | ChronosBolt_small_finetuned_full_lr1e-5 - ARIMA_AICc             |  61 |   -1.7986 | 0      |   0      |
| snl       |        5 | ChronosBolt_small_finetuned_full_lr1e-5 - ChronosBolt_small_zero |  61 |   -2.112  | 0      |   0      |
| snl       |        8 | ChronosBolt_small_finetuned - hybrid_sqrt_xgb                    |  39 |    1.3366 | 0.0004 |   0.0012 |
| snl       |        8 | ChronosBolt_small_finetuned - ARIMA_AICc                         |  39 |   -0.1341 | 0.2767 |   0.5534 |
| snl       |        8 | ChronosBolt_small_finetuned - ChronosBolt_small_zero             |  39 |   -1.9202 | 0      |   0.0001 |
| snl       |        8 | ChronosBolt_small_finetuned_full_lr1e-5 - hybrid_sqrt_xgb        |  39 |    1.3825 | 0.0003 |   0.0012 |
| snl       |        8 | ChronosBolt_small_finetuned_full_lr1e-5 - ARIMA_AICc             |  39 |   -0.0882 | 0.4265 |   0.5534 |
| snl       |        8 | ChronosBolt_small_finetuned_full_lr1e-5 - ChronosBolt_small_zero |  39 |   -1.8744 | 0      |   0.0001 |
| luh       |       10 | ChronosBolt_small_finetuned - hybrid_sqrt_xgb                    | 104 |    0.3133 | 0.0138 |   0.0138 |
| luh       |       10 | ChronosBolt_small_finetuned - ARIMA_AICc                         | 104 |   -0.7434 | 0.0052 |   0.0105 |
| luh       |       15 | ChronosBolt_small_finetuned - hybrid_sqrt_xgb                    | 104 |    0.2663 | 0.7828 |   0.7828 |
| luh       |       15 | ChronosBolt_small_finetuned - ARIMA_AICc                         | 104 |   -0.2614 | 0.2025 |   0.405  |
| luh       |       20 | ChronosBolt_small_finetuned - hybrid_sqrt_xgb                    | 104 |    0.2604 | 0.1924 |   0.1924 |
| luh       |       20 | ChronosBolt_small_finetuned - ARIMA_AICc                         | 104 |   -0.3185 | 0.0072 |   0.0144 |

### G7 full fine-tune (lr 1e-5) selected checkpoints

| dataset   |   fold |   step |    val |
|:----------|-------:|-------:|-------:|
| snl       |      1 |    300 | 3.456  |
| snl       |      2 |    200 | 4.4651 |
| snl       |      3 |    200 | 2.4039 |
| snl       |      4 |    100 | 2.2987 |
| snl       |      5 |    100 | 3.5697 |

## 7. Uncertainty and decisions


### Luh hybrid split-conformal (independent calibration cells)

|   origin |   nominal |   coverage_equal_cell |   mean_width_Ah |   mean_width_SOH |   finite_fraction |   fitonly_point_MAPE |
|---------:|----------:|----------------------:|----------------:|-----------------:|------------------:|---------------------:|
|       10 |       0.8 |                0.8362 |          0.2779 |           0.0942 |                 1 |               2.767  |
|       10 |       0.9 |                0.9464 |          0.497  |           0.1685 |                 1 |               2.767  |
|       15 |       0.8 |                0.8237 |          0.2469 |           0.0837 |                 1 |               2.5858 |
|       15 |       0.9 |                0.9122 |          0.4606 |           0.1561 |                 1 |               2.5858 |
|       20 |       0.8 |                0.7914 |          0.2297 |           0.0779 |                 1 |               2.5136 |
|       20 |       0.9 |                0.9006 |          0.4881 |           0.1655 |                 1 |               2.5136 |

### Luh group-conditional (regime) conformal, overall

|   origin |   nominal |   coverage |   width_SOH |   finite |
|---------:|----------:|-----------:|------------:|---------:|
|       10 |       0.8 |     0.8083 |      0.0848 |   0.9922 |
|       10 |       0.9 |     0.9029 |      0.1326 |   0.8807 |
|       15 |       0.8 |     0.7901 |      0.0775 |   0.9894 |
|       15 |       0.9 |     0.9073 |      0.1242 |   0.8308 |
|       20 |       0.8 |     0.794  |      0.0856 |   0.9831 |
|       20 |       0.9 |     0.9211 |    inf      |   0.678  |

### Luh group-conditional conformal by regime

|   origin |   nominal |   age_type |   coverage |   width_SOH |   finite |
|---------:|----------:|-----------:|-----------:|------------:|---------:|
|       10 |       0.8 |          1 |     0.8214 |      0.0363 |   1      |
|       10 |       0.8 |          2 |     0.7311 |      0.114  |   1      |
|       10 |       0.8 |          3 |     0.9068 |      0.147  |   0.9615 |
|       10 |       0.9 |          1 |     0.8978 |      0.0598 |   0.936  |
|       10 |       0.9 |          2 |     0.8776 |      0.1783 |   0.8546 |
|       10 |       0.9 |          3 |     0.9569 |      0.2228 |   0.798  |
|       15 |       0.8 |          1 |     0.8003 |      0.0331 |   1      |
|       15 |       0.8 |          2 |     0.7369 |      0.1167 |   1      |
|       15 |       0.8 |          3 |     0.8554 |      0.1138 |   0.9473 |
|       15 |       0.9 |          1 |     0.9043 |      0.054  |   0.9118 |
|       15 |       0.9 |          2 |     0.8803 |      0.1793 |   0.7851 |
|       15 |       0.9 |          3 |     0.9594 |      0.1931 |   0.7219 |
|       20 |       0.8 |          1 |     0.8056 |      0.0291 |   1      |
|       20 |       0.8 |          2 |     0.7604 |      0.1425 |   1      |
|       20 |       0.8 |          3 |     0.8239 |      0.1197 |   0.9161 |
|       20 |       0.9 |          1 |     0.9182 |    inf      |   0.8542 |
|       20 |       0.9 |          2 |     0.9214 |    inf      |   0.5929 |
|       20 |       0.9 |          3 |     0.9271 |    inf      |   0.4173 |

### Sandia hybrid CV+-style intervals

| family          |   nominal |   coverage_equal_cell |   mean_width_SOH_pp |    LFP |    NCA |    NMC |   origin |
|:----------------|----------:|----------------------:|--------------------:|-------:|-------:|-------:|---------:|
| hybrid_sqrt_xgb |       0.8 |                0.7433 |              3.0919 | 0.8646 | 0.3735 | 0.8113 |        5 |
| hybrid_sqrt_xgb |       0.9 |                0.868  |              4.8979 | 0.929  | 0.5949 | 0.9618 |        5 |
| hybrid_sqrt_xgb |       0.8 |                0.7565 |              3.5396 | 0.7974 | 0.4    | 0.7573 |        8 |
| hybrid_sqrt_xgb |       0.9 |                0.8506 |              5.2064 | 0.8801 | 0.5407 | 0.867  |        8 |

### Luh decision metrics (common script)

| method                 |   origin |   true_events |   precision |   recall |     F1 |   last_APE |   signed_bias_pct |   optimistic_share |   abs_cross_step_err |
|:-----------------------|---------:|--------------:|------------:|---------:|-------:|-----------:|------------------:|-------------------:|---------------------:|
| ARIMA_AICc             |       10 |            54 |      0.9565 |   0.8148 | 0.88   |     9.8578 |            3.149  |             0.4327 |               1.8636 |
| ARIMA_AICc             |       15 |            54 |      1      |   0.8704 | 0.9307 |     9.0804 |            5.7951 |             0.5385 |               0.6809 |
| ARIMA_AICc             |       20 |            54 |      0.963  |   0.963  | 0.963  |     6.4583 |            4.2544 |             0.4904 |               0.2885 |
| avg_ARIMA_XGB          |       10 |            54 |      0.9259 |   0.9259 | 0.9259 |     7.0785 |            0.4598 |             0.3558 |               2.3667 |
| avg_ARIMA_XGB          |       15 |            54 |      0.9392 |   0.9506 | 0.9448 |     6.8115 |            2.7606 |             0.4519 |               0.6948 |
| avg_ARIMA_XGB          |       20 |            54 |      0.9522 |   0.9815 | 0.9666 |     4.9235 |            1.9322 |             0.3846 |               0.3585 |
| avg_ARIMA_XGB_NN       |       10 |            54 |      0.9451 |   0.9568 | 0.9509 |     7.4909 |            2.2907 |             0.375  |               2.5548 |
| avg_ARIMA_XGB_NN       |       15 |            54 |      0.9691 |   0.9444 | 0.9561 |     6.823  |            3.5729 |             0.4519 |               0.8366 |
| avg_ARIMA_XGB_NN       |       20 |            54 |      0.9463 |   0.9753 | 0.9605 |     4.9484 |            1.5195 |             0.4327 |               0.3924 |
| avg_ARIMA_hybrid       |       10 |            54 |      0.963  |   0.963  | 0.963  |     6.7927 |            2.6569 |             0.3462 |               1.8205 |
| avg_ARIMA_hybrid       |       15 |            54 |      0.9625 |   0.9506 | 0.9565 |     6.7504 |            4.2288 |             0.3942 |               0.7532 |
| avg_ARIMA_hybrid       |       20 |            54 |      0.9579 |   0.9815 | 0.9695 |     4.5853 |            2.9285 |             0.3846 |               0.3522 |
| best_neural            |       10 |            54 |      0.9461 |   0.7531 | 0.8381 |    11.8168 |            5.9526 |             0.6731 |               2.7213 |
| best_neural            |       15 |            54 |      0.9576 |   0.8333 | 0.8911 |     9.07   |            5.1973 |             0.6923 |               0.963  |
| best_neural            |       20 |            54 |      0.9309 |   0.9136 | 0.922  |     7.3919 |            0.6942 |             0.5673 |               0.5068 |
| gru__history_only      |       20 |            54 |      0.9692 |   0.9691 | 0.9691 |     5.5267 |            2.3937 |             0.4423 |               0.3057 |
| hybrid_sqrt_xgb        |       10 |            54 |      0.9461 |   0.9753 | 0.9605 |     5.2152 |            2.1649 |             0.4423 |               2.4241 |
| hybrid_sqrt_xgb        |       15 |            54 |      0.9522 |   0.9815 | 0.9666 |     5.3571 |            2.6626 |             0.3558 |               1.0629 |
| hybrid_sqrt_xgb        |       20 |            54 |      0.9464 |   0.9815 | 0.9636 |     3.9367 |            1.6025 |             0.375  |               0.478  |
| persistence            |       10 |            54 |      1      |   0.0741 | 0.1379 |    18.8752 |           18.8752 |             1      |               0      |
| persistence            |       15 |            54 |      1      |   0.463  | 0.6329 |    14.7946 |           14.7946 |             1      |               0      |
| persistence            |       20 |            54 |      1      |   0.6296 | 0.7727 |    10.3714 |           10.3714 |             1      |               0      |
| regime_origin_selector |       10 |            54 |      0.962  |   0.9383 | 0.9499 |     7.2275 |            3.6012 |             0.6154 |               2.4276 |
| regime_origin_selector |       15 |            54 |      0.9628 |   0.9568 | 0.9598 |     6.8689 |            3.388  |             0.4519 |               0.8129 |
| regime_origin_selector |       20 |            54 |      0.933  |   0.9444 | 0.9387 |     6.2483 |            1.6163 |             0.4327 |               0.4314 |
| rf                     |       10 |            54 |      0.9091 |   0.9259 | 0.9174 |     6.99   |           -1.351  |             0.4615 |               2.7533 |
| rf                     |       15 |            54 |      0.9177 |   0.963  | 0.9398 |     6.9516 |           -0.017  |             0.3269 |               1.0128 |
| rf                     |       20 |            54 |      0.9354 |   0.9815 | 0.9578 |     4.6282 |           -0.4355 |             0.3173 |               0.478  |
| sqrt_k_lag1            |       10 |            54 |      0.9091 |   0.9259 | 0.9174 |     8.6469 |            6.0097 |             0.7115 |               3.2    |
| sqrt_k_lag1            |       15 |            54 |      0.9259 |   0.9259 | 0.9259 |     8.5952 |            6.7508 |             0.5577 |               1.42   |
| sqrt_k_lag1            |       20 |            54 |      0.9434 |   0.9259 | 0.9346 |     8.0428 |            6.7817 |             0.5385 |               0.7    |
| xgb                    |       10 |            54 |      0.9    |   0.8889 | 0.8944 |     7.111  |           -2.2294 |             0.4231 |               3.0903 |
| xgb                    |       15 |            54 |      0.9133 |   0.9753 | 0.9433 |     6.1266 |           -0.2738 |             0.4327 |               1.0949 |
| xgb                    |       20 |            54 |      0.8988 |   0.9815 | 0.9382 |     4.869  |           -0.3901 |             0.3462 |               0.4528 |

### Sandia hybrid decision metrics

| family          |   last_step_APE |   last_step_signed_pct |   share_optimistic |   crossing_F1 |   true_crossing_cells |   mean_abs_crossing_step_error |   origin |
|:----------------|----------------:|-----------------------:|-------------------:|--------------:|----------------------:|-------------------------------:|---------:|
| hybrid_sqrt_xgb |          1.7272 |                 0.4896 |             0.5246 |        0.6896 |                    21 |                           0.65 |        5 |
| hybrid_sqrt_xgb |          2.0969 |                 0.7654 |             0.5385 |        0.2051 |                     8 |                           0    |        8 |

## 8. Robustness


### Noise / dropped-checkup robustness at origin 20 (round-3 perturbations)

|   sigma |   drop |   ARIMA_AICc |   compact_lstm |    gru |   large_lstm |   persistence |     rf |   sqrt_idx_lag1 |    xgb |   hybrid |
|--------:|-------:|-------------:|---------------:|-------:|-------------:|--------------:|-------:|----------------:|-------:|---------:|
|   0     |   0    |       2.761  |         3.9797 | 3.8825 |       3.5494 |        4.8771 | 2.7259 |          4.6401 | 2.7228 |   2.1821 |
|   0     |   0.1  |       2.6351 |         3.9865 | 3.8592 |       3.5754 |        4.8771 | 2.7587 |          4.6227 | 2.713  |   2.3171 |
|   0     |   0.25 |       2.6451 |         4.0193 | 3.7983 |       3.5886 |        4.8771 | 2.88   |          4.5659 | 2.74   |   2.7268 |
|   0.005 |   0    |       3.9901 |         4.0769 | 4.0258 |       3.5753 |        4.8908 | 3.1338 |          4.6708 | 2.931  |   2.9012 |
|   0.005 |   0.1  |       3.9326 |         4.0839 | 4.0054 |       3.6007 |        4.8908 | 3.111  |          4.6563 | 2.9339 |   2.9961 |
|   0.005 |   0.25 |       3.9828 |         4.1089 | 3.9636 |       3.6055 |        4.8908 | 3.1712 |          4.6035 | 2.9813 |   3.2162 |
|   0.01  |   0    |       4.3778 |         4.2404 | 4.2532 |       3.6862 |        5.0037 | 3.6572 |          4.7419 | 3.6055 |   3.6911 |
|   0.01  |   0.1  |       4.3423 |         4.2447 | 4.2408 |       3.7041 |        5.0037 | 3.6197 |          4.73   | 3.5985 |   3.7126 |
|   0.01  |   0.25 |       4.4066 |         4.2763 | 4.2019 |       3.7265 |        5.0037 | 3.7019 |          4.6828 | 3.5897 |   3.7223 |

### Hybrid minus ARIMA under perturbation

|   sigma |   drop |   n |   hybrid_minus_ARIMA |      p |   p_holm |
|--------:|-------:|----:|---------------------:|-------:|---------:|
|   0     |   0    | 104 |              -0.5789 | 0.1763 |   0.3526 |
|   0     |   0.1  | 104 |              -0.318  | 0.7189 |   0.7189 |
|   0     |   0.25 | 104 |               0.0817 | 0.1135 |   0.3406 |
|   0.005 |   0    | 104 |              -1.0889 | 0      |   0.0001 |
|   0.005 |   0.1  | 104 |              -0.9366 | 0      |   0.0004 |
|   0.005 |   0.25 | 104 |              -0.7666 | 0.0039 |   0.027  |
|   0.01  |   0    | 104 |              -0.6867 | 0.0398 |   0.2387 |
|   0.01  |   0.1  | 104 |              -0.6297 | 0.0669 |   0.2677 |
|   0.01  |   0.25 | 104 |              -0.6844 | 0.0521 |   0.2604 |

### Knee vs non-knee cells

| dataset   | method          |   origin |   knee_cells |   MAPE_knee |   MAPE_nonknee |
|:----------|:----------------|---------:|-------------:|------------:|---------------:|
| Luh       | ARIMA_AICc      |       10 |            1 |      4.2631 |         3.2714 |
| Luh       | ARIMA_AICc      |       15 |            1 |      6.4648 |         2.6764 |
| Luh       | ARIMA_AICc      |       20 |            1 |      2.8823 |         2.7598 |
| Luh       | avg_ARIMA_XGB   |       10 |            1 |      6.3206 |         3.1766 |
| Luh       | avg_ARIMA_XGB   |       15 |            1 |      4.0941 |         2.4854 |
| Luh       | avg_ARIMA_XGB   |       20 |            1 |      1.4973 |         2.2943 |
| Luh       | hybrid_sqrt_xgb |       10 |            1 |      2.5476 |         2.2211 |
| Luh       | hybrid_sqrt_xgb |       15 |            1 |      4.9975 |         2.1578 |
| Luh       | hybrid_sqrt_xgb |       20 |            1 |      4.0842 |         2.1636 |
| Luh       | sqrt_k_lag1     |       10 |            1 |      6.4744 |         3.1612 |
| Luh       | sqrt_k_lag1     |       15 |            1 |      9.3662 |         3.0904 |
| Luh       | sqrt_k_lag1     |       20 |            1 |     22.4504 |         4.3155 |
| Sandia    | ARIMA_AICc      |        5 |            4 |      8.2275 |         4.3115 |
| Sandia    | hybrid_sqrt_xgb |        5 |            4 |      1.9132 |         1.4749 |
| Sandia    | sqrt_efc_lag1   |        5 |            4 |      6.3751 |         2.1867 |
| Sandia    | ARIMA_AICc      |        8 |            4 |      7.3864 |         2.6929 |
| Sandia    | hybrid_sqrt_xgb |        8 |            4 |      3.303  |         1.5207 |
| Sandia    | sqrt_efc_lag1   |        8 |            4 |      9.1811 |         1.8426 |
| Kirkaldy  | ARIMA_AICc      |        3 |            4 |     21.4728 |         8.7838 |
| Kirkaldy  | ARIMA_AICc      |        5 |            4 |     22.0165 |         5.8637 |
| Kirkaldy  | ARIMA_AICc      |        8 |            4 |     13.4341 |         2.7163 |
| Kirkaldy  | hybrid_transfer |        3 |            4 |     14.0791 |         2.9708 |
| Kirkaldy  | hybrid_transfer |        5 |            4 |     15.619  |         2.2704 |
| Kirkaldy  | hybrid_transfer |        8 |            4 |     15.1269 |         2.0845 |
| Kirkaldy  | persistence     |        3 |            4 |     21.4728 |         8.7838 |
| Kirkaldy  | persistence     |        5 |            4 |     22.0165 |         6.2015 |
| Kirkaldy  | persistence     |        8 |            4 |     22.2528 |         4.503  |
| Kirkaldy  | sqrt_idx_prior  |        3 |            4 |     16.4852 |         3.1406 |
| Kirkaldy  | sqrt_idx_prior  |        5 |            4 |     18.3583 |         2.8094 |
| Kirkaldy  | sqrt_idx_prior  |        8 |            4 |     23.2701 |         2.6536 |

## Pending / not available
- none