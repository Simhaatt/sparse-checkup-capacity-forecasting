# Kirkaldy sparse checkup schema, version 1

This adapter reads the five official experiment archives from Zenodo record
[10637534](https://zenodo.org/records/10637534), the accompanying
`experiment_metadata.xlsx`, and three compact summaries per cell. It does not
re-extract capacities from raw timeseries. All 40 metadata cells must resolve to
exactly one performance, ageing-set, and cycle summary. Raw files remain unchanged.

## Canonical mapping

| Output | Source / definition | Availability and meaning |
|---|---|---|
| `cell_id` | `KIR_E{experiment}_{cell letter}`; comma in 2,2 becomes underscore | Experiment plus cell, not cell letter alone |
| `condition_id` | Experiment plus temperature setpoint | 15 protocol/temperature groups; no claim that rigs and channels are identical |
| `checkup_idx` | Performance `Ageing Sets` | RPT identifier, not cycle number |
| `row_id` | Cell plus RPT | Unique canonical key |
| `source_record_idx` | Zero-based CSV data record before sorting | Blank lines ignored; not a physical text line number |
| `rpt_date` | `RPT start date` | Explicit ISO or DD/MM/YYYY parsing; timezone and sub-day precision unavailable |
| `elapsed_days`, `delta_days` | RPT date minus RPT0 date; successive difference | Real checkup-to-checkup wall-clock time, first increment zero |
| `source_days_of_degradation` | `Days of degradation`, unchanged | Source-specific clock; never substituted for wall-clock time |
| `Q_Ah` | `C/10 Capacity [mA h]` / 1000 | Measured C/10 discharge capacity; no fallback to fitted `Cell Capacity` |
| `Q0_Ah`, `SOH_Q0` | RPT0 capacity, Q/Q0 | Reference available within every permitted context; no target-fitted scaler |
| `source_cell_capacity_Ah`, `source_soh` | Original `Cell Capacity` and `SoH` | Retained for audit, not the primary label |
| `operating_cycles` | `Ageing Cycles` | Protocol cycle count, not EFC |
| `temperature_setpoint_C` | Metadata `Temp` | Imposed condition, separate from measured summary |
| `age_set_temperature_mean_C` | Performance average ageing temperature | Covers ageing sets, not necessarily the entire interval including RPT/rest |
| `soc_min_scenario`, `soc_max_scenario` | Numeric metadata SoC range / 100 | Imposed limits; drivecycle remains missing, no invented 0–100 mapping |
| `charge_C`, `discharge_C` | Missing | Compact summaries do not supply validated current/time rate summaries |
| `source_throughput_Ah`, `source_energy_throughput_Wh` | Performance summary cumulative values | Source-labelled; ageing-only versus full-lifetime coverage needs reconciliation |
| `source_delta_throughput_Ah`, `source_delta_energy_Wh` | Successive differences | Historical summaries; future values require an explicit scenario/oracle protocol |
| `EFC_Q0_proxy`, `delta_EFC_Q0_proxy` | Throughput/(2Q0), difference | Explicit exploratory proxy, not the frozen common EFC |
| `delta_EFC` | Missing | Prevents accidental matching to Luh's nominal-capacity EFC/total coverage |
| `set_source_*` | Every ageing-set column, joined by exact set ID | Original alternative exposure estimates retained without overwriting performance values |
| `R_0p1s_source_Ohm_aux` | `0.1s Resistance [Ohms]` | Auxiliary only; absent on many odd checkups |

Each row also carries archive/member names, member SHA256 hashes, and the metadata
SHA256. The source manifest hashes all 120 consumed members, including cycle
summaries retained for subsequent exposure reconciliation. ZIP reads verify CRC.
This run does not recompute the SHA256 of all 49 GB of original archives.

## Scientific limits and permitted use

The currently validated common core is observed capacity/SOH history and elapsed
checkup time. Capacity test rates differ by source and must remain in protocol
metadata; dimensionless SOH does not make the protocols identical. Temperature,
SOC, and exposure features require availability masks and explicit semantics.
Reported ageing-set means must not be labelled full BMS interval telemetry.

The cycle summaries provide evidence that the field called charge throughput
includes charge and discharge contributions: the first rows sum both directions.
However, whole-sequence cumulative sums do not exactly reproduce reported totals,
and performance versus set summaries disagree for some cells. No corrected EFC
or fully harmonized exposure feature is asserted here. The nominal 5 Ah versus
observed Q0 denominator also requires an explicit cross-dataset decision.

No fitted target-cell normalization, monotonic clipping, interpolation, model
training, or test-based model selection occurs. The separate cumulative-minimum
table is sensitivity-only. A cell is evaluable at context K only when n > K, so
there is at least one forecast target. The external split manifest describes
held-out experiments; adaptation pools still require an inner group split.

## Outputs and checks

The package contains raw CSV/Parquet checkups, a separate monotonic sensitivity
Parquet, cell QC, K=3/5/10 populations, missingness, exposure discrepancy and
monotonicity audits, archive inventory, source and output checksums, leave-one-
experiment-out assignments, code, metadata, and a completion receipt.

Duplicate keys, ambiguous file discovery, missing RPT0, invalid capacity,
unparseable/non-increasing RPT dates, negative cumulative exposure increments,
and mismatched set keys fail explicitly. Source-to-source disagreements are
reported rather than silently reconciled. No capacity-increase rows are removed.

Resumption hashes the current adapter, metadata, and consumed source members.
Cached Parquet contents must match a stored SHA256 before reuse. Changed code or
input creates a new dataset identity. Final files are written atomically and a
checksum/completion record is written after all outputs. Cache directory is not
part of the downloadable processed package.

## Run

Local: `python kirkaldy_preprocess.py --input-root "F:\battery dataset"`.

Kaggle: upload `kirkaldy_compact_sources.zip` as an input dataset and run
`Kirkaldy_Preprocessing_v1.ipynb`. The compact package contains the same consumed
member bytes inside five small ZIPs bearing the original archive names, plus
metadata. It needs CPU only. The corrected notebook supports intact nested ZIPs,
an intact outer compact bundle, and recursively extracted experiment folders.
Extracted folders are identified from exact performance-summary filenames,
independent of the dataset slug. It reconstructs only compact summaries under
the writable output directory and marks them as reconstructed in the manifests.
Their CRC values describe the reconstructed ZIP; SHA256 hashes describe the
consumed file contents. The original archive inventory is supplied
in that package for provenance; reduced ZIPs are explicitly marked as repackaged.

The notebook embeds the entire adapter, so it needs no workspace files. Resume a
Kaggle session by restoring its `cell_cache` directory and selecting the same
output root. CSV exports accompany Parquet for inspection; Parquet preserves types.
