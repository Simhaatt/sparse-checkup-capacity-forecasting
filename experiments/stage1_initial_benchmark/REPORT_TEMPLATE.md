# JES evidence round 3: capacity-only experiment handoff

This is an experiment report and evidence handoff. It does not draft or edit the manuscript. Its completed status is valid only with the numerical verification and visual inspection receipts linked below. Individual job records, checkpoint hashes and environment receipts identify the actual runs; the source protocol was frozen during 6–7 October 2026.

## Completion and requested scope

| Requested item | Evidence in this report |
| --- | --- |
| 1. Equal-budget retuning | 500 search fits: 20 configurations × five model families × five folds. Inner-validation selection, three-seed refits, quiet inference measurements, training costs and validation frontiers. |
| 2. Information ablations | Full F7, no operation counter, prefix-static descriptors and capacity history only at origin 20. Static/no-counter inputs are identical in this corpus; explicitly labelled checkpoint aliases avoid claiming separate fits. |
| 3. Fair descriptor selection | Only predeclared variants; fitting, scaling and checkpoint selection inside the training partitions. The old global feature ranking is retired. |
| 4. Raw capacity audit | The three specified raw cells, direct capacity reconstruction and a past-only cumulative-minimum check. |
| 5. Raw resistance alignment | All 228 raw EIS files obtained and CRC checked; completed-spectrum grouping and a backward join with a fixed 14-day tolerance. Missing labels remain missing. |
| 6. Resistance arms A/B/C | Same cells, folds, seeds and selected configurations; B/C have identical input dimensions and predict capacity only. The aligned post-origin joint-macro diagnostic is unavailable, as expressly accepted by the user. |
| 7. Practitioner checks | A CSV checklist covering sweep identification, clocks, temporal availability, missingness and separate endpoints. |
| 8. Combinations and selector | Validation-selected neural representative, equal-weight triple and all pairwise averages, and an inner-validation-only ageing-regime/origin selector. |
| 9. Prediction intervals | Horizon-specific 80% and 90% split-conformal intervals from separate calibration cells; coverage, width and finite-interval fraction overall and by ageing type. |
| 10. Decision metrics and horizon errors | Last-step error, signed bias, optimistic predictions, precisely defined 80%-health crossings and error-versus-horizon curves. |
| 11. Pretrained baselines | Official pinned Chronos-Bolt small and tiny weights; zero-shot median forecasts and native central 80% quantiles. |
| 12. Population ageing law | Training-cell Arrhenius/power-law fits and one prefix-only cell calibration scale, using a corrected nominal full-equivalent-cycle proxy. |
| 13. Learning curves | Tuned XGBoost and the validation-selected neural model at 25%, 50% and 100% fitting-cell fractions, three random subset repeats and three model seeds. |
| Earlier request: second dataset | Independent Oxford eight-cell checkup campaign; eight held-out-cell folds, five learned families and simple/pretrained controls. |
| Earlier request: robustness | Shared synthetic noise and missing-prefix scenarios for all 104 cells; 2,600 ARIMA fits and 75 learned-model inference jobs, with unchanged future targets. |

## Instruction authority and preserved history

The user's requests define the work. [The pasted round-3 request](audit/USER_ROUND3_REQUEST.txt) supplies the latest detailed protocol; attachments provide reference material. The user requested experiments and a comprehensive Markdown handoff, and expressly excluded manuscript drafting. Earlier work and supplied files remain preserved. No GitHub push, DOI creation or journal contact was performed.

The earlier 100-fit exploratory search used a different split and compact-model range. It remains historical evidence outside this primary comparison. Earlier frozen clean-capacity results are a secondary reference. Rejected resistance reconstruction and exploratory population-law versions are retained and labelled; they do not supply the current scientific results.

The earlier requested XGBoost transfer arms and square-root-fade plus learned-residual transfer study are already completed on 40 target cells at the declared measurement budgets. Their full-precision summaries and paired tests are retained below as earlier-round evidence. They are not replaced by the separate Oxford study. Read [the preserved earlier report](earlier_round/REPORT_HISTORICAL.md) with the contributor-trace corrections in this report; its earlier blanket calendar/profile timestamp inference is superseded.

The original multi-objective optimizer A/B runs were not executed on Kaggle. The previously accepted fallback is a short historical candidate-generator role, with no active hypervolume or parity claims. The new 500-fit corrected-trainer random search supplies the primary comparison; it is not mislabelled as execution of the old multi-objective notebooks. Earlier selection used loaded timings, as disclosed in that round. The current quiet measurements independently provide validation frontiers without treating the old contaminated selection as a fresh quiet re-selection.

## Dataset, partitions and reproducible model protocol

The source comparison scores the same 104 eligible cells in the supplied five grouped outer folds. A cell requires more than 20 checkups for this cohort. Eligibility is retrospective; this does not establish prospective performance on cells excluded by trajectory length.

`Q0` is the first processed capacity observation, not automatically a manufacturer's nominal rating. Source targets are the supplied capacity trajectories after a past-only cumulative minimum. The benchmark therefore assesses processed capacity degradation rather than untouched measurement noise. The raw-input audit and synthetic-noise study have separate, explicitly limited roles.

Each fold reserves 27 selection-validation cells: the rounded 15% of all outer-training cells, sampled among eligible cells with seed `1000 + fold`. Twenty additional, disjoint calibration cells are withheld. Actual fitting pools contain 135 or 136 cells. Calibration cells do not select configurations, stop training or choose selector rules. This smaller fitting pool differs from the earlier frozen benchmark; a before/after difference is not an isolated causal estimate of retuning benefit.

Each of compact LSTM, GRU, large LSTM, XGBoost and random forest receives 20 predeclared random configurations per fold. Compact LSTM uses one layer and hidden widths 16–64; large LSTM uses widths 128–256 and one or two layers. The complete spaces, draws and schedules are in [PLAN.json](PLAN.json). Search seed is 42; final seeds are 42, 52 and 62. Matching trial counts does not match CPU hours. The subsequent best-neural choice compares three family winners, representing 60 neural candidate opportunities, rather than one tree family's 20.

Neural fitting comprises teacher-forced pretraining followed by corrected recursive fine-tuning. Both stages select checkpoints using equal-cell validation capacity error. Output predictions are mapped back to the input scaler's coordinates before feedback. Scalers use fitting cells only. Final configurations are selected by recursive inner-validation capacity MAPE, with deterministic trial-order tie breaking.

Origin 20 is primary. Origins 10 and 15 replay origin-20-selected structures with history length clipped to the available prefix; they are not new origin-specific searches. Inputs for target row `i` contain rows `i-L` through `i-1`. After the forecast origin, capacity is self-fed. Full F7 consists of operation counter, temperature, charge/discharge rates, SOC-window descriptor, ageing type and initial capacity, alongside capacity history. Full F7 is conditional on the supplied operating schedule. Reduced variants omit the future operation counter. All six remaining descriptors are constant within each source cell, making no-counter and prefix-static inputs exactly identical; their 75 static records are aliases of fitted no-counter models, with zero additional fitting cost. This is numerical input identity, not a statistical equivalence conclusion.

## Estimands, statistical tests and practical decisions

Primary capacity MAPE averages stepwise absolute percentage errors within each seed/cell, then averages seed errors within each cell, then weights cells equally. It does not compute error after averaging predictions across training seeds. Ensembles average member predictions within each seed before scoring. Confidence intervals use 10,000 cell bootstrap draws with seed 20261005. Paired Wilcoxon tests use the explicitly named Holm families in the linked tables; non-significance is not equivalence.

The equal-weight combinations use ARIMA, tuned XGBoost and the inner-selected neural model. All three pairs and the triple are scored. The selector's ageing-regime/origin rules use selection-validation results only, with a pooled-validation fallback; outer-test performance does not select the rule.

Conformal scores are absolute capacity residuals in Ah on the independent calibration cells. At each method/fold/seed/horizon, the radius is order statistic `ceil((n+1) × nominal coverage)`. If this rank exceeds the available count, the interval is unbounded. Lower capacity bounds are truncated at zero. Tables report coverage, Ah/SOH widths, finite fraction and horizon counts. Coverage is marginal per horizon under exchangeability, not simultaneous trajectory coverage or a guaranteed ageing-type-conditional rate. The observed cohort changes at long horizons.

The 80%-health threshold is `0.8 × Q0`. Event assessment includes cells whose last measured prefix capacity is at least that threshold. A crossing is the first future value strictly below it. Absence of a crossing is censored at `H+1`; separate errors are supplied for jointly observed crossings and censored timing. Precision, recall and F1 are computed by training seed and then averaged, with event counts supplied. Last-step APE, mean signed percentage bias and the fraction of optimistic predictions are separately reported. Error-versus-horizon tables weight available cells equally after averaging seed errors.

## Raw electrochemical provenance and corrections

The direct raw EOC capacity column is retained through the EIS join. The cumulative minimum uses only preceding retained capacity observations. Exact reconstruction on the three specified cells does not independently audit the upstream capacity estimator or every raw cell.

Raw `seq_nr` can be a frequency-point counter reused across checkups. The old heuristic sometimes grouped a frequency across dates instead of grouping a complete spectrum. The corrected reconstruction groups points sharing a completed-event timestamp, applies the specified room-temperature/SOC-50/validity restrictions and averages the three highest-frequency compensated real parts in mΩ. It uses neither whole-cell clipping nor a cumulative maximum. The first new reconstruction attempt also used invalid sequence groups; it is explicitly rejected in the audit. Only the version-2 aligned stream is current.

The fixed backward 14-day join yields exactly one matched capacity observation per source cell, all before origin 10. All 228 raw files were obtained, so this sparsity is an availability finding, not failed acquisition. Arm C tests initial resistance information plus missingness; it cannot establish evolving resistance forecasts. Missing raw labels stay NaN. A numerical zero with mask zero is only an input encoding. Initial resistance becomes available only after its first causal arrival. Following both evaluation origins and simulated training-segment origins, resistance and its initial value are held from pre-origin information, and the mask is zero. B and C have equal channel counts; B also has a prefix availability mask.

Actual contributor traces correct an earlier blanket lookahead inference. The raw initial-resistance means for the three audited cells use pre-origin measurements at origins 10 and 20. A group's latest timestamp alone did not identify the three contributors to the mean. The cyclic cell's resistance history does contain future contributors: 9 of 10 prefix rows at origin 10 and 19 of 20 at origin 20, with maximum lookahead approximately 294.65 and 84.72 days. Calendar/profile contributor traces do not show that same defect at those origins. Invalid grouping, whole-cell transforms and full-group Rct-like statistics remain separate issues. Differences between B and C compare the reconstructed streams as a whole; they do not isolate one operator or validate B for deployment.

The user explicitly chose to keep the fixed rule and report the aligned post-origin joint-macro diagnostic as unavailable. [The decision is preserved](audit/USER_ALIGNMENT_DECISION.json). Missing resistance error is not replaced with zero, no join is widened or moved forward, and no claim of physical resistance forecasting follows from this study. Historical joint capacity/resistance scores are labelled as such and excluded from the active capacity ranking.

## Population law, transfer, learning and robustness limits

The population comparator has the form

`Q/Q0 = 1 - alpha_cell × [a_calendar(T,SOC) × t_previous^z + a_cycle(T,C-rate,DoD) × nominalFEC_previous^w]`.

Arrhenius terms and power exponents are fitted on training cells only, with four training-loss restart candidates per seed. One bounded, nonnegative cell scale is calibrated on the measured prefix; there is no per-cell intercept. The law is empirical and inspired by semi-empirical ageing formulations; it is not a reproduction of published coefficients or mechanistic parameter identification. SOC windows obtained from raw operating voltage setpoints add metadata beyond F7, so its information is not identical to the learned-family comparison.

On profile cells, the supplied operation counter counts partial discharge events rather than full cycles. It remains unchanged as a benchmark identifier. The population law instead uses completed normal-operation recharge counts multiplied by nominal protocol depth of discharge, relative to the first checkup, excluding diagnostic cycles. This avoids capacity-derived future Ah throughput. It is a nominal exposure proxy with fixed-SOC-window and partial-recharge limitations. A profile discharge-rate field of zero lacks a waveform-average estimate; it is not evidence of physically zero discharge current.

Learning curves use nested, shared random subsets drawn only from fitting cells. Three subset repeats and three model seeds are used at 25% and 50%; the main full-pool results supply the 100% reference. Validation/calibration counts remain fixed. Configurations were tuned with the full fitting pool, so these are conditional fixed-structure curves, not evidence that an entire campaign or search requires only the displayed subset size. No extra subsampling variance is claimed at 100%.

Oxford uses raw 1C discharge endpoint charge differences, without a cumulative minimum. One exact copied diagnostic in Cell 1 was removed. Observed timestamps are MATLAB serial days despite generic README wording about seconds. Each of eight held-out-cell folds fits six cells and validates on the next cyclic cell; source-fold-1-selected structures are frozen before Oxford evaluation. Learned histories use `Q/Q0` only, with scalers fitted on the six training cells. No Oxford test result selects a configuration. Exact chemistry is not asserted from the README alone; eight cells cannot establish broad cross-chemistry generality.

Robustness adds Gaussian prefix noise of 0%, 0.5% or 1% of Q0 and drops 0%, 10% or 25% of checkups. The first reference and last prefix observation are retained. Perturbations and masks are shared across methods; non-clean cases use three random repeats. Learned predictors use causal capacity imputation; ARIMA refits the observed perturbed prefix on the missing-observation grid. Future targets are unchanged. These are prespecified synthetic sensitivities, not an empirically estimated field-noise distribution, and learned models are not retrained on noisy outer-test data.

Chronos-Bolt small and tiny use pinned official weights in zero-shot mode on the same measured prefixes. Native 10th/90th percentiles support central 80% intervals; native 90% intervals are not fabricated from those tails. Conformal 90% intervals are separately calibrated. Pretraining overlap is not independently audited. The models forecast in checkup-index time despite irregular elapsed spacing.

## Resource accounting and interpretation

Training CPU and wall times, serialized payload size, neural parameter counts or tree-node counts, and inference timing are supplied. Quiet measurements use one CPU thread, actual inner-validation histories representing available ageing regimes, warm-up calls, repeated blocks and three measurement passes. They are core model-call measurements with preprocessing excluded and differing neural/tree APIs. Ambient OS load is not fully isolated, and these laptop numbers are not embedded-BMS latency guarantees.

Frontiers use validation error, quiet latency and serialized size within each fold; a four-objective sensitivity adds fitting CPU time. The cross-fold mean frontier is descriptive. Outer-test error does not choose frontier membership. Serialized payloads include scalers where applicable; tree nodes and neural parameters are distinct size measures.

## Numerical tables and negative findings

Tables below print six significant figures; linked CSVs retain full precision. Negative, non-significant, sparse-label and unbounded-interval findings remain in the evidence. The full catalogue supplies additional per-cell, per-seed, configuration, horizon, resource and audit detail.

