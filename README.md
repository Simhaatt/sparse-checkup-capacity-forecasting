# Forecasting lithium-ion capacity from sparse checkups

Code, processed data and result artifacts for the manuscript

> Simhaa T. T. and B. Sivakumar, *Forecasting lithium-ion capacity from sparse checkups: a prior-anchored residual learner that transfers across cell formats*, submitted to the **Journal of Energy Storage** (2026).

The method fits a square-root capacity-fade prior to each cell's measured checkups. A gradient-boosted residual learner, trained across other cells at simulated forecast origins, then corrects the prior's future error. A fixed 0.5/0.5 average with a per-cell ARIMA forecast is also evaluated. Four public datasets are used:

- **Luh–Blank** (228 cells) is the primary benchmark.
- **Sandia** (80 cells) is the independent multi-chemistry benchmark.
- **Kirkaldy 21700** (40 cells) and **Oxford pouch** (8 cells) are zero-target transfer cohorts.

The repository also documents the leakage audit that removed resistance from the benchmark.

## Quick start: Level-1 reproduction (about 15 s on a laptop CPU, no GPU, no raw data)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce/level1.py
```

Expected last line: `OVERALL: PASS`. A machine-readable report is written to `results/reproduction_check.json`.

Level 1 starts from the deposited test-set predictions and does the following:

1. It runs the **original analysis scripts unchanged**: `r4_analyse.py`, `r4_decisions_v2.py` and `r4_g.py g5`.
2. It recomputes every other summary table with `summarise_from_predictions.py`.
3. It compares the 19 regenerated tables with the deposited tables. The tolerance is 1e-9.
4. It checks 68 headline numbers quoted in the manuscript.
5. It re-checks the values plotted in Figs. 3–6.

## Level 2: re-running the experiments

This needs the raw datasets and several CPU-hours; the neural searches benefit from a GPU. See [`docs/RUN_ORDER.md`](docs/RUN_ORDER.md) for the download → preprocessing → experiment → analysis order, and [`data/README.md`](data/README.md) for data sources.

## Repository layout

| Folder | Contents |
|---|---|
| `data/processed/` | Processed capacity tables redistributed under the source licences (see [`DATA_LICENSE.md`](DATA_LICENSE.md)), plus cell-grouped fold assignments |
| `data/sandia/` | Sandia cell metadata only; the Sandia cycle data must be downloaded from batteryarchive.org |
| `preprocessing/` | Download and preprocessing code per dataset (Luh–Blank, Sandia check-up extraction, Kirkaldy, Oxford) |
| `leakage_audit/` | Resistance-alignment audit: contributor trace of the faulty join, corrected backward 14-day join, raw-EIS check |
| `experiments/stage0_capacity_retuning/` | Earlier capacity-only experiments: nested search, zero-shot Chronos, Oxford preparation, semi-empirical controls |
| `experiments/stage1_initial_benchmark/` | Initial grouped benchmark (manuscript: "initial grouped benchmark"): ARIMA, √ prior, RF, XGBoost, LSTM/GRU, combinations, robustness |
| `experiments/stage2_sandia/` | Independent Sandia benchmark (check-up extraction output → grouped folds → models) |
| `experiments/stage3_main_study/` | Main study: residual learner, origin-specific re-tuning, transfer, ablations, intervals, decisions, robustness; analysis-plan file `PLAN_R4.json` and run logs |
| `results/predictions/` | Cell-level test (and validation/calibration) predictions behind every reported number (gzip CSV) |
| `results/tables/` | All summary tables: `main_study/`, `initial_benchmark/`, `sandia/` (aggregates only) |
| `paper_figures/` | Figure scripts, the exact plotted tables (`data_used/`), editable draw.io diagrams, and PDF/SVG/PNG exports of Figs. 1–6 and A.1 |
| `reproduce/` | Level-1 script and the list of manuscript values it checks |
| `docs/` | Run order, code provenance, and the auto-generated main-study report |

[`MANUSCRIPT_MAP.md`](MANUSCRIPT_MAP.md) maps every manuscript table and figure to the producing script and file.

## Provenance and honesty notes

- **Code is unchanged from the runs.** The experiment folders hold the scripts exactly as they were run. The only edits are in `stage3_main_study`, `stage2_sandia` and `paper_figures/scripts/prepare_plot_data.py`, which used absolute paths on the original machines; these were made relative or configurable. Each edited line is marked `# repo edit`. See [`docs/CODE_PROVENANCE.md`](docs/CODE_PROVENANCE.md).
- **Test cells never influenced a choice.** Model selection used inner-validation cells only. The analysis plan and its time-stamped amendments are in `experiments/stage3_main_study/PLAN_R4.json`. Internal plans are not a public preregistration.
- **The origin-20 re-tuning was never run.** It was dropped under a recorded compute amendment, and no value is reported for it.
- **The Sandia origin-8 fine-tuned Chronos result is not confirmatory.** Its checkpoint selection reused validation cells.
- **Resistance is not used.** An earlier processed derivative of Luh–Blank joined impedance (EIS) sweeps to capacity rows by nearest row number. This attached future resistance to past rows. A corrected backward 14-day join leaves one resistance value per cell, so the benchmark is capacity-only. The leaky columns (`Re`, `Rct`, `Re0`, `Rct0`) are still present in `data/processed/luh_blank/phase1_cv_all_rows.csv` so the audit can be checked. They must not be used as forecast inputs.

## Data and licences

| Dataset | Source | Licence | In this repository |
|---|---|---|---|
| Luh & Blank (2024), 228 NMC/C-SiO 18650 cells | [doi:10.35097/1969](https://doi.org/10.35097/1969); Sci. Data 11, 1004 | CC BY 4.0 | processed capacity table and predictions |
| Kirkaldy et al. (2024), 40 LG M50T 21700 cells | [doi:10.5281/zenodo.10637534](https://doi.org/10.5281/zenodo.10637534) | CC BY 4.0 | processed feature table and predictions |
| Oxford Battery Degradation Dataset 1, 8 Kokam pouch cells | [doi:10.5287/bodleian:KO2kdmYGg](https://doi.org/10.5287/bodleian:KO2kdmYGg) | ODbL 1.0 | processed table and predictions, **under ODbL** |
| Sandia / Preger et al. (2020), 86 18650 cells | [batteryarchive.org](https://www.batteryarchive.org) | no explicit licence stated | **not redistributed**: metadata, code and aggregate results only |

Details and attribution are in [`DATA_LICENSE.md`](DATA_LICENSE.md). The code is MIT-licensed ([`LICENSE`](LICENSE)).

## Citation

Please cite the manuscript and this archive; see [`CITATION.cff`](CITATION.cff). The Zenodo DOI of each release is shown on the Zenodo record linked from the release page.

## Contact

Simhaa T. T., Sri Sivasubramaniya Nadar College of Engineering, Chennai, India. Please use GitHub issues for questions about the code.
