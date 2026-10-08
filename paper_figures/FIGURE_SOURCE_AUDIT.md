# Figure source audit

**Principal numerical authority:** `02_ALL_RESULTS.md` (JES writing pack). It was generated from the round-4 tables, and the same tables sit unmodified in `data_used/source_tables/`.

**Automated checks:**

- `scripts/check_figure_values.py` asserts every check value given in the figure brief: **29 of 29 passed**.
- `scripts/prepare_plot_data.py` asserts that every derived mean equals the published summary.

## Figure 3: primary accuracy (`fig3_primary_accuracy.csv`)

| Plotted item | Source | Check |
|---|---|---|
| Means and 95 % CIs: square-root prior (`sqrt_k_lag1`), ARIMA, residual learner, ARIMA–residual average, round-3 ARIMA–XGB average, re-tuned best neural | `r4_summary.csv` (copied unchanged) | ARIMA 3.28 / 2.71 / 2.76; residual learner 2.22 / 2.19 / 2.18; average 2.35 / 2.14 / 2.07: **pass** |
| Fine-tuned Chronos-Bolt-small means | `r4_chronos_ft_summary.csv` (2.54 / 2.45 / 2.44) | recomputed from the saved predictions and asserted equal |
| Fine-tuned Chronos 95 % CIs | **derived for this figure** from `r4_chronos_ft_luh_pred.csv`, with the round-4 procedure (cell bootstrap, 10,000 resamples, `default_rng(20261005)`, percentile) | the summary table has no CIs, so these are computed, not invented |
| Re-tuned best neural at origin 20 | absent from every table, because the R4A compute amendment dropped origin 20 | drawn as "not run" |
| Significance asterisks | `r4_pairs.csv`, families `r4_vs_ARIMA_origin{10,15,20}` | marked only where Holm p < 0.05: average at 10 (1.0e-8), 15 (4.5e-5) and 20 (5.3e-4); residual learner at 10 (1.6e-3). The residual learner at 15 (0.29) and 20 (0.18) is not marked. |

## Figure 4: forecast horizon (`fig4_error_by_horizon.csv`)

| Plotted item | Source | Check |
|---|---|---|
| Equal-cell APE per step: ARIMA, residual learner, average | Recomputed at full precision from the saved test predictions (`r4_analyse.load`). Asserted equal to the 2-decimal `r4_hybrid_error_by_horizon.csv`. | At origin 10, step 1: ARIMA 0.21, residual learner 0.56. Step 19: ARIMA 6.53, residual learner 2.67, average 3.78. **pass** |
| Cells per step | counted from the saved predictions | 104 at step 1; 81 at the final step for every origin. **pass** |

The curves are plotted point for point. They are not smoothed or interpolated, and no horizon-level CIs are drawn.

## Figure 5: transfer (`fig5_transfer.csv`)

| Plotted item | Source | Check |
|---|---|---|
| Means | `r4_transfer_summary.csv` (cylindrical) and `r4_g1_oxford_summary.csv` (pouch) | Residual learner, cylindrical: 4.08 / 3.61 / 3.39. Pouch: 4.40 / 5.07 / 1.87. **pass**. Recomputed means asserted equal to the summaries. |
| 95 % CIs | **derived for this figure** from `r4_transfer_kirkaldy_pred.csv` and `r4_g1_oxford_pred.csv`, with the same bootstrap procedure | computed from verified predictions |

## Figure 6: intervals and decisions

| Panel | Source | Notes |
|---|---|---|
| (a) | `r4_hybrid_conformal_luh.csv` | Coverage at 80 % nominal: 0.836 / 0.824 / 0.791. At 90 %: 0.946 / 0.912 / 0.901. Finite fraction 1.0. |
| (b) | `r4_g5_mondrian_by_regime.csv`, 80 % nominal | The regime coding (1 calendar, 2 cyclic, 3 profile) was verified from the data. Profile finite fraction is 0.96 / 0.95 / 0.92; unbounded intervals count as covering. 90 % rows are not plotted, because their finite fractions are as low as 0.42. |
| (c) | `r4_hybrid_conformal_snl.csv`, 80 % nominal; cell counts from `r4_hybrid_snl_pred.csv` | These are CV+-style cross-fold residual intervals, not split-conformal. The panel title says so. NCA: n = 13 at origin 5 and n = 3 at origin 8, printed in the bars. |
| (d) | `r4_decisions_luh_r3rule.csv` (**the corrected file**; the older `r4_decisions_luh.csv` is not used) | ARIMA vs residual learner F1: 0.87 vs 0.96 (origin 10), 0.87 vs 0.94 (origin 15), 0.90 vs 0.90 (origin 20). **pass**. Crossings / eligible cells: 50/100, 30/80, 20/70. |

## Diagrams (Figures 1, 2 and A.1)

| Figure | Facts used | Source |
|---|---|---|
| 1 | 228 cells; 5 grouped folds; per fold 135–136 fitting, 27 validation, 20 calibration, 45–46 test; 104 scored cells with more than 20 checkups; refit on fit + calibration (R4B rule); conformal uses a fit-only model with independent calibration | 01_PAPER_DETAILS §4.1, §6.2, §8.2; `audit/cell_partitions.csv` |
| 2 | lag-one √ prior by OLS, slope ≤ 0, target prefix only; XGBoost trained on other cells at simulated origins with targets SOH − prior; inference features (step, origin, prior, last SOH, last and mean residual, slope, gap, planned schedule); per-cell ARIMA (AICc); fixed 0.5 / 0.5 average | 01_PAPER_DETAILS §6; `r4_hybrid.py` |
| A.1 | EIS ordinals of about 1–28 vs raw capacity rows in the thousands; nearest join with no tolerance; cyclic cell 9 of 10 rows (≈ 295 days) at origin 10 and 19 of 20 (≈ 85 days) at origin 20; calendar and profile cells: no actual future contributors; corrected backward 14-day join leaves one value per cell, all before origin 10 | 01_PAPER_DETAILS §5.1; dossier B §A.9 |

Fig. A.1 is conceptual: its marker positions are schematic, and the figure says so ("time (schematic)"). Fig. 1b is a table of information sources and contains no data. Fig. 2 is a signal-flow diagram and contains no data.

## Visual QA performed

Every PDF was rendered and inspected at reading size, as was the contact sheet. Problems found and fixed during QA:

- clipped legends and axis labels
- a panel label overlapping a title
- overlapping annotations in Fig. A.1
- revisions 3–4: the three diagrams were redrawn from scratch (revision 5: engineering block-diagram style, side-by-side panels); checked at reading size for crossings between connectors and labels and for clipping at the canvas edge (graph bounds within the 1000 px canvas)

Specific checks:

- **Legends:** all placed outside the plotting areas.
- **Line and marker sizes:** plot lines 2.5 pt; error bars 2–2.5 pt; markers 7.5–11 pt.
- **Colours:** consistent per model in every figure. Residual learner blue, ARIMA orange, average aqua, √ prior grey, round-3 ARIMA–XGB violet, fine-tuned Chronos pink, re-tuned neural green.
- **Palette checks:** the dataviz validator passed CVD and normal-vision separation. Lower-contrast slots also carry distinct marker shapes and dark edges.
- **PDF output:** vector only (`pdfimages` reports no images), with fonts embedded.
