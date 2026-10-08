# Data licences and attribution

The source code in this repository is released under the MIT licence (`LICENSE`). The **data files** listed below are derived from third-party datasets and keep the terms of those datasets. They are not covered by the MIT licence.

## Luh–Blank (CC BY 4.0)

Files:

- `data/processed/luh_blank/*`
- all files in `results/predictions/` except `results/predictions/oxford/`
- Luh–Blank rows in `results/tables/`

Source: M. Luh, T. Blank, *Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell*, Scientific Data 11 (2024) 1004, [doi:10.1038/s41597-024-03831-x](https://doi.org/10.1038/s41597-024-03831-x). Dataset: RADAR4KIT, [doi:10.35097/1969](https://doi.org/10.35097/1969), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes made:

- checkup capacities were extracted from end-of-charge records with a past-only cumulative minimum;
- capacity was normalised by the first checkup;
- operating descriptors were added;
- forecasts were produced by the models in this repository.

The resistance columns (`Re`, `Rct`, `Re0`, `Rct0`) come from an earlier processed derivative whose impedance alignment was found to be faulty (see `leakage_audit/`). They are kept only to document that audit.

## Kirkaldy et al. (CC BY 4.0)

Files:

- `data/processed/kirkaldy/*`
- `results/predictions/r4_transfer_*`
- Kirkaldy rows in `results/tables/`

Sources:

- Dataset: N. Kirkaldy, M. A. Samieian, G. J. Offer, M. Marinescu, Y. Patel, *Data from: Lithium-ion battery degradation: comprehensive cycle ageing data and analysis for commercial 21700 cells*, Zenodo, [doi:10.5281/zenodo.10637534](https://doi.org/10.5281/zenodo.10637534), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Article: J. Power Sources 603 (2024) 234185.

Changes made:

- the prepared 40-cell feature table was built from the source data;
- capacity was normalised by the first checkup;
- the exposure column normalised by the final checkup is not used.

## Oxford Battery Degradation Dataset 1 (ODbL 1.0)

Files: `data/processed/oxford/*` and `results/predictions/oxford/*`.

Source: D. Howey, C. Birkl, *Oxford Battery Degradation Dataset 1*, University of Oxford Research Archive, [doi:10.5287/bodleian:KO2kdmYGg](https://doi.org/10.5287/bodleian:KO2kdmYGg), licensed under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).

These derived files are themselves made available under the **ODbL v1.0**. If you redistribute or adapt them, you must keep this notice and offer your adapted database under the ODbL.

Changes made:

- capacity was taken from the 1C discharge charge difference and converted from mAh to Ah;
- one exactly duplicated record was removed;
- forecasts were added.

## Sandia National Laboratories (not redistributed)

Source: Y. Preger et al., *Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions*, J. Electrochem. Soc. 167 (2020) 120532, [doi:10.1149/1945-7111/abae37](https://doi.org/10.1149/1945-7111/abae37). The data are distributed by [batteryarchive.org](https://www.batteryarchive.org). Its code of conduct asks users to cite the article and batteryarchive.org, but no redistribution licence is stated.

**What this repository contains for Sandia:**

- the cell metadata (`data/sandia/snl_cells_metadata.csv`), i.e. the published test conditions;
- the check-up extraction code;
- the fold assignments (`results/tables/sandia/origin*/partitions.csv`);
- aggregate results.

**What it does not contain:** Sandia cycle data, extracted capacities or cell-level predictions.

To reproduce the Sandia results, follow `data/README.md`.
