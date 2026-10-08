# Figure manifest

Panel sub-captions "(a) …" sit **below** each panel. All PDFs are 180 mm wide, vector, with fonts embedded (Liberation Sans Bold; the "✕" glyph falls back to DejaVu Sans) and no raster images.

| # | PDF (`figures/`) | Size (mm) | Method | Editable source | Data |
|---|---|---|---|---|---|
| 1 | `fig_forecast_protocol.pdf` | 180 × 105 | draw.io | `editable/fig_forecast_protocol.drawio` | partition counts (01_PAPER_DETAILS §4.1) |
| 2 | `fig_residual_method.pdf` | 180 × 82 | draw.io | `editable/fig_residual_method.drawio` | method description (01_PAPER_DETAILS §6; `r4_hybrid.py`) |
| 3 | `fig_primary_accuracy.pdf` | 180 × 124 | Python | `scripts/fig_primary_accuracy.py` | `fig3_primary_accuracy.csv`, `fig3_holm_vs_arima.csv` |
| 4 | `fig_forecast_horizon.pdf` | 180 × 104 | Python | `scripts/fig_forecast_horizon.py` | `fig4_error_by_horizon.csv` |
| 5 | `fig_external_transfer.pdf` | 180 × 106 | Python | `scripts/fig_external_transfer.py` | `fig5_transfer.csv` |
| 6 | `fig_interval_decisions.pdf` | 180 × 214 | Python | `scripts/fig_interval_decisions.py` | `fig6_*.csv` |
| A.1 | `fig_appendix_leakage_timeline.pdf` | 180 × 112 | draw.io | `editable/fig_appendix_leakage_timeline.drawio` | audit facts (01_PAPER_DETAILS §5.1) |

## Suggested captions and labels

I could not open `main.tex` in this session, so the labels below are suggestions. Keep your own labels if they already exist.

**Fig. 1** `\label{fig:protocol}`

> Forecasting evaluation protocol.
>
> (a) Cell-grouped evaluation of the primary dataset. The 228 cells are split into five outer folds by cell identity. Each fold has 135–136 fitting, 27 validation, 20 calibration and 45–46 test cells. The 104 cells with more than 20 checkups form the common scoring cohort, and each is tested once.
>
> Point forecasts select settings on the validation cells and are then refitted on fitting and calibration cells. Split-conformal intervals use a model fitted on fitting cells only, so that the calibration cells stay independent.
>
> (b) Forecast information boundary. Columns are checkups up to the forecast origin o and after it; blue cells are model inputs and grey cells are excluded. Inputs are limited to capacity measured up to the forecast origin, fixed cell descriptors and the planned operating schedule. Post-origin capacity is used only for scoring; resistance and realised throughput after the origin are never inputs, and the benchmark does not use them before the origin either.

**Fig. 2** `\label{fig:method}`

> Prior-anchored residual forecasting.
>
> - **Prior:** a square-root fade law is fitted by ordinary least squares to the target cell's measured prefix (slope b ≤ 0), giving the prior p(i). It is an empirical ageing reference, not a mechanistic model.
> - **Residual learner:** an XGBoost model with parameters θ, trained offline on other cells at simulated forecast origins with target SOH − p, predicts the residual r̂(i) from prefix features z(i) and the prior.
> - **Forecasts:** the residual-learner forecast is ŷ₁(i) = p(i) + r̂(i), and ŷ₂(i) is a per-cell ARIMA forecast (orders by AICc). The final forecast is ŷ(i) = 0.5 ŷ₁(i) + 0.5 ŷ₂(i).

**Fig. 3** `\label{fig:accuracy}`

> Capacity forecasting error on the 104-cell cohort at forecast origins of 10, 15 and 20 checkups. Error bars are 95 % cell-bootstrap confidence intervals (10,000 resamples).
>
> (a) Square-root prior, ARIMA, residual learner and the fixed ARIMA–residual average.
>
> (b) Additional comparators: the round-3 ARIMA–XGBoost average, fine-tuned Chronos-Bolt-small, and the best neural model re-tuned at each origin. Re-tuning at origin 20 was not run.
>
> \* Holm-adjusted p < 0.05 versus ARIMA within the pre-declared family. The residual learner alone is not significantly better than ARIMA at origins 15 and 20.

**Fig. 4** `\label{fig:horizon}`

> Equal-cell capacity error by future checkup step after (a) 10, (b) 15 and (c) 20 measured checkups. ARIMA is most accurate one step ahead; the residual learner's error grows more slowly with horizon. Values are as measured, with no smoothing. The number of contributing cells falls from 104 at step 1 to 81 at the final step.

**Fig. 5** `\label{fig:transfer}`

> Cross-format transfer.
>
> (a) 40 cylindrical 21700 cells at origins 3, 5 and 8.
>
> (b) 8 pouch cells at origins 5, 10 and 20.
>
> The frozen unit-free residual learner was trained without any target-dataset records: on Luh–Blank, Sandia and Oxford for (a), and on Luh–Blank and Sandia only for (b). Each target cell still supplies its own measured prefix at forecast time. Error bars are 95 % cell-bootstrap confidence intervals.

**Fig. 6** `\label{fig:intervals}`

> Prediction intervals and threshold decisions for the residual learner.
>
> (a) Primary dataset: empirical coverage of split-conformal 80 % and 90 % intervals, calibrated on 20 independent cells per fold. Dashed lines mark nominal coverage.
>
> (b) Regime-stratified (Mondrian) 80 % intervals for calendar, cyclic and driving-profile cells. Some profile intervals are unbounded because calibration groups are small (finite fraction 0.92–0.96).
>
> (c) Independent benchmark (Sandia): 80 % coverage of cross-fold residual intervals by chemistry. These are not split-conformal. Bars show cell counts; NCA has only 3 cells at origin 8.
>
> (d) F1 score for detecting the first crossing below 80 % of initial capacity, using the round-3 eligibility rule.

**Fig. A.1** `\label{fig:leakage}`

> Resistance-alignment audit (conceptual timeline).
>
> (a) In the audited processed derivative, EIS sweeps carried short ordinal indices while capacity rows kept raw row indices. A nearest-number join without a time tolerance attached later resistance values to earlier capacity records. For the audited cyclic cell, 9 of 10 resistance-prefix rows at origin 10 used future EIS (maximum about 295 days ahead), and 19 of 20 at origin 20 (about 85 days). The audited calendar and profile cells showed no actual future contributors.
>
> (b) Corrected matching: complete spectra, chronological backward-only joins within 14 days, and missing values kept missing. This yields one matched resistance value per cell, all before origin 10, so the benchmark uses capacity only.
