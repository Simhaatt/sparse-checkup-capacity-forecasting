# Data

Level-1 reproduction needs nothing beyond this repository. Level 2 needs the source datasets below, obtained from their publishers under their own terms (see `../DATA_LICENSE.md`).

## Processed tables in this repository

| File | Rows | Description |
|---|---|---|
| `processed/luh_blank/phase1_cv_all_rows.csv` | 3,980 checkups, 228 cells | Checkup-level capacity series with operating descriptors. `Q` is the checkup capacity in Ah; `Q0` is the first checkup; `age_type` is 1 calendar, 2 cyclic, 3 driving profile; `k_exp` is the exposure coordinate. **`Re`, `Rct`, `Re0` and `Rct0` come from the faulty impedance alignment (see `leakage_audit/`) and are not used by any reported model.** |
| `processed/luh_blank/grouped_cv_folds.csv` | 228 | Outer-fold assignment by cell |
| `processed/luh_blank/cell_partitions.csv` | 228 × 5 folds | Per-fold role of each cell (fit / validation / calibration / test) |
| `processed/kirkaldy/kirkaldy_40_normalized_features.csv` | 511 checkups, 40 cells | Kirkaldy reference-checkup capacities and descriptors (the exposure normalised by the final checkup is not used) |
| `processed/oxford/oxford_capacity_checkups.csv` | 8 cells | Oxford 1C diagnostic discharge capacities (ODbL) |
| `sandia/snl_cells_metadata.csv` | 86 cells | Published test conditions of the Sandia cells (no measurements) |

## Source datasets (Level 2)

1. **Luh–Blank** (CC BY 4.0), [doi:10.35097/1969](https://doi.org/10.35097/1969).
   - Download the result-data archive from RADAR4KIT.
   - `preprocessing/luh_blank/verified_preprocessing_cell11.py` is the checkup-extraction code that produced `phase1_cv_all_rows.csv`. It is cell 11 of the original preprocessing notebook.
   - `preprocessing/luh_blank/audit_raw_capacity.py` reconstructs three representative cells from raw end-of-charge records and confirms they match exactly.
   - `download_luh_compact.py` fetches the compact cycling-log member that was used in the exposure audit.
   - These are historical scripts: set the local paths at the top of each file.
2. **Kirkaldy et al.** (CC BY 4.0), [doi:10.5281/zenodo.10637534](https://doi.org/10.5281/zenodo.10637534).
   - Download the Zenodo record.
   - Run `python preprocessing/kirkaldy/kirkaldy_preprocess.py --input-root <folder>`.
   - Then run `validate_kirkaldy_preprocessing.py`. Expected source hashes are in `kirkaldy_delivery_sha256.json`.
3. **Oxford Battery Degradation Dataset 1** (ODbL), [doi:10.5287/bodleian:KO2kdmYGg](https://doi.org/10.5287/bodleian:KO2kdmYGg).
   - Run `python preprocessing/oxford/fetch_external.py oxford`, or download `Oxford_Battery_Degradation_Dataset_1.mat` to `external/oxford/`.
   - Then run `python preprocessing/oxford/prepare_oxford.py`.
4. **Sandia** (Preger et al. 2020), distributed by [batteryarchive.org](https://www.batteryarchive.org) (SNL study).
   - Export the cycle-level data of the 86 SNL 18650 cells into one CSV, `snl_cycle_data.csv`, with the batteryarchive cycle-table fields `cell_id, cycle_index, test_time, ah_c, ah_d, e_c, e_d`. Use the cell identifiers in `sandia/snl_cells_metadata.csv`.
   - Then run `python preprocessing/sandia/extract_checkups.py` from a folder that contains `snl_cycle_data.csv` and `snl_cells_metadata.csv`. It writes `snl_checkups.csv`: 80 retained cells, with 61 eligible at origin 5 and 39 at origin 8.
   - Place `snl_checkups.csv` in `experiments/`; the Sandia benchmark reads `../snl_checkups.csv`.
   - Please cite Preger et al. (2020) and batteryarchive.org, as the archive's code of conduct asks.
