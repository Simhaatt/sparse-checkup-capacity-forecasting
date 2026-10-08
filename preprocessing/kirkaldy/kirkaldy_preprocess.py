"""Kirkaldy 10637534 sparse checkup adapter; reads only compact ZIP members.

Run: python kirkaldy_preprocess.py --input-root "F:\\battery dataset"
Dependencies: pandas, numpy, openpyxl, pyarrow. No raw archives are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import platform
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "kirkaldy-sparse-v1"
EXPERIMENTS = ["1", "2,2", "3", "4", "5"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode())


def write_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temp, index=False)
    else:
        frame.to_csv(temp, index=False)
    temp.replace(path)


def parse_dates(values):
    # Source experiment 5 uses DD/MM/YYYY, others use ISO dates.
    def one(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        s = str(v).strip()
        return pd.to_datetime(s, format="%d/%m/%Y" if "/" in s else "%Y-%m-%d", errors="raise")
    return values.map(one).astype("datetime64[ns]")


def discover(root, filename):
    found = sorted(root.rglob(filename))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {filename} below {root}; found {found}")
    return found[0]


def member(z, suffix):
    matches = [n for n in z.namelist() if n.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"Expected one member ending {suffix}: {matches}")
    return matches[0]


def resolve_experiment(root, long_name, exp, base):
    """Accept original/nested ZIPs or Kaggle's recursively extracted summaries."""
    archives = sorted(root.rglob(f"{long_name}.zip"))
    if len(archives) > 1:
        raise ValueError(f"Multiple experiment ZIPs: {archives}")
    if archives:
        return archives[0], "input_zip"
    # Anchor on an exact known member, independent of dataset slug and enclosing folders.
    anchors = sorted(root.rglob(f"Expt {exp} - cell A (10degC) - Processed Data.csv"))
    anchors = [p for p in anchors if p.parent.name == "Performance Summary"]
    if len(anchors) != 1:
        raise ValueError(f"Expected one ZIP or extracted Performance Summary for experiment {exp}; "
                         f"found {len(archives)} ZIPs and {len(anchors)} folder anchors below {root}. "
                         "Attach one complete kirkaldy_compact_sources dataset.")
    summary_root = anchors[0].parent.parent
    folders = [summary_root / "Performance Summary",
               summary_root / "Ageing Sets Summary" / "Summary per Set",
               summary_root / "Ageing Sets Summary" / "Summary per Cycle"]
    if not all(p.is_dir() for p in folders):
        raise ValueError(f"Incomplete extracted summary layout: {summary_root}")
    # Reconstruct only the compact members in writable output; never touch Kaggle inputs.
    archive = base / "reconstructed_inputs" / f"{long_name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            for p in sorted(folder.glob("*.csv")):
                name = long_name + "/Summary Data/" + p.relative_to(summary_root).as_posix()
                z.write(p, name)
    temp.replace(archive)
    print(f"Discovered extracted summaries for experiment {exp}: {summary_root}", flush=True)
    return archive, "reconstructed_from_extracted_summaries"


def read_csv(data):
    # The official files contain a literal Unicode replacement character in temperature headers.
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def convert(perf, sets, meta, provenance):
    d = perf.copy()
    d["source_record_idx"] = np.arange(len(d))  # zero-based parsed data record, blank lines ignored
    d["Ageing Sets"] = pd.to_numeric(d["Ageing Sets"], errors="raise")
    if d["Ageing Sets"].duplicated().any() or not np.equal(d["Ageing Sets"] % 1, 0).all():
        raise ValueError("Duplicate/non-integer checkup identifiers")
    d = d.sort_values("Ageing Sets", kind="stable").reset_index(drop=True)
    if d.iloc[0]["Ageing Sets"] != 0:
        raise ValueError("Missing RPT0: do not silently replace the allowed reference")
    if sets["Age Set"].duplicated().any():
        raise ValueError("Duplicate ageing-set keys")
    if set(d["Ageing Sets"]) != {0, *sets["Age Set"].tolist()}:
        raise ValueError("Performance/set key mismatch")
    dates = parse_dates(d["RPT start date"])
    q = pd.to_numeric(d["C/10 Capacity [mA h]"], errors="raise") / 1000
    if not np.isfinite(q).all() or not (q > 0).all() or dates.isna().any():
        raise ValueError("Missing/non-positive capacity or missing RPT date")
    if not (dates.diff().dropna() > pd.Timedelta(0)).all():
        raise ValueError("Non-increasing checkup dates")
    out = pd.DataFrame({"checkup_idx": d["Ageing Sets"].astype(int), "source_record_idx": d["source_record_idx"]})
    for k, v in provenance.items():
        out[k] = v
    out["dataset"] = "kirkaldy_10637534"
    out["cell_id"] = f"KIR_E{str(meta['Expt']).replace(',', '_')}_{meta['Cell']}"
    out["experiment_id"] = str(meta["Expt"])
    out["condition_id"] = f"KIR_E{str(meta['Expt']).replace(',', '_')}_T{int(meta['Temp'])}"
    out["row_id"] = out["cell_id"] + "_RPT" + out["checkup_idx"].astype(str)
    out["rpt_date"] = dates
    out["rpt_date_raw"] = d["RPT start date"]
    out["age_set_start_date"] = parse_dates(d["Age set start date"])
    out["age_set_end_date"] = parse_dates(d["Age set end date"])
    out["elapsed_days"] = (dates - dates.iloc[0]).dt.total_seconds() / 86400
    out["delta_days"] = out["elapsed_days"].diff().fillna(0)
    out["source_days_of_degradation"] = d["Days of degradation"]
    out["Q_Ah"] = q
    out["Q0_Ah"] = float(q.iloc[0])
    out["SOH_Q0"] = q / q.iloc[0]
    out["capacity_protocol"] = "RPT_C_over_10_discharge"
    out["source_cell_capacity_Ah"] = d["Cell Capacity [mA h]"] / 1000
    out["source_soh"] = d["SoH"]
    out["delta_Q_Ah"] = q.diff()
    out["raw_capacity_increase"] = out["delta_Q_Ah"] > 0
    out["operating_cycles"] = d["Ageing Cycles"]
    out["temperature_setpoint_C"] = float(meta["Temp"])
    temp_cols = [c for c in d if c.startswith("Age set av. temperature [")]
    if len(temp_cols) != 1:
        raise ValueError("Unexpected temperature header")
    out["age_set_temperature_mean_C"] = d[temp_cols[0]]
    out["soc_protocol_raw"] = str(meta["SoC range"])
    soc = re.fullmatch(r"(\d+)-(\d+)", str(meta["SoC range"]))
    out["soc_min_scenario"] = int(soc[1]) / 100 if soc else np.nan
    out["soc_max_scenario"] = int(soc[2]) / 100 if soc else np.nan
    out["charge_C"] = np.nan
    out["discharge_C"] = np.nan
    out["source_throughput_Ah"] = d["Charge Throughput [A h]"]
    out["source_energy_throughput_Wh"] = d["Energy Throughput [W h]"]
    out["source_delta_throughput_Ah"] = out["source_throughput_Ah"].diff().fillna(0)
    out["source_delta_energy_Wh"] = out["source_energy_throughput_Wh"].diff().fillna(0)
    if (out[["source_delta_throughput_Ah", "source_delta_energy_Wh"]] < 0).any().any():
        raise ValueError("Negative source cumulative exposure increment")
    out["EFC_Q0_proxy"] = out["source_throughput_Ah"] / (2 * q.iloc[0])
    out["delta_EFC_Q0_proxy"] = out["EFC_Q0_proxy"].diff().fillna(0)
    # Leave the strict common field unavailable until throughput coverage is harmonized.
    out["delta_EFC"] = np.nan
    out["exposure_status"] = "source_summary_proxy_not_full_BMS_interval"
    out["R_0p1s_source_Ohm_aux"] = d["0.1s Resistance [Ohms]"]
    sets = sets.copy()
    sets["set_source_record_idx"] = np.arange(len(sets))
    cols = {c: "set_source_" + c for c in sets if c != "Age Set"}
    sets = sets.rename(columns=cols)
    out = out.merge(sets, left_on="checkup_idx", right_on="Age Set", how="left", validate="one_to_one").drop(columns="Age Set")
    out["throughput_summary_difference_Ah"] = out["source_throughput_Ah"] - out["set_source_Cumulative Charge Throughput [A h]"]
    out["energy_summary_difference_Wh"] = out["source_energy_throughput_Wh"] - out["set_source_Cumulative Energy Throughput [W h]"]
    out["throughput_summary_mismatch"] = out["throughput_summary_difference_Ah"].abs() > 1e-4
    out["energy_summary_mismatch"] = out["energy_summary_difference_Wh"].abs() > 1e-3
    return out


def run(input_root, output_root):
    root, base = Path(input_root), Path(output_root)
    metadata = discover(root, "experiment_metadata.xlsx")
    meta_sha = file_sha(metadata)
    code_sha = file_sha(__file__)
    tables, sources, inventory, fingerprints = [], [], [], []
    cache = base / "cell_cache"
    for exp in EXPERIMENTS:
        meta = pd.read_excel(metadata, sheet_name=f"expt {exp}")
        if meta["Cell"].duplicated().any():
            raise ValueError("Duplicate metadata cell key")
        long_name = meta.iloc[0]["Expt long name"]
        archive, source_layout = resolve_experiment(root, long_name, exp, base)
        with zipfile.ZipFile(archive) as z:
            actual = [n for n in z.namelist() if "/Performance Summary/" in n and n.endswith(".csv")]
            if len(actual) != len(meta):
                raise ValueError("Metadata/summary cell count mismatch")
            for info in z.infolist():
                inventory.append({"archive": archive.name, "source_layout": source_layout, "member": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"})
            for _, m in meta.iterrows():
                cell = m["Cell"]
                p_name = member(z, f"/Performance Summary/Expt {exp} - cell {cell} ({int(m['Temp'])}degC) - Processed Data.csv")
                s_name = member(z, f"/Summary per Set/expt {exp} - cell {cell} - set_data.csv")
                c_name = member(z, f"/Summary per Cycle/expt {exp} - cell {cell} - cycle_data.csv")
                raw = {n: z.read(n) for n in [p_name, s_name, c_name]}
                provenance = {"source_archive": archive.name, "source_member": p_name, "source_member_sha256": sha(raw[p_name]), "set_source_member": s_name, "set_source_member_sha256": sha(raw[s_name]), "metadata_sha256": meta_sha}
                fingerprint = sha(json.dumps([VERSION, code_sha, meta_sha, [(n, sha(b)) for n, b in raw.items()]], sort_keys=True).encode())
                fingerprints.append(fingerprint)
                part = cache / f"{fingerprint}.parquet"
                receipt = part.with_suffix(".json")
                if part.exists() and receipt.exists() and json.loads(receipt.read_text())["sha256"] == file_sha(part):
                    out = pd.read_parquet(part)
                    status = "resumed"
                else:
                    out = convert(read_csv(raw[p_name]), read_csv(raw[s_name]), m, provenance)
                    write_table(part, out)
                    write_json(receipt, {"sha256": file_sha(part), "fingerprint": fingerprint})
                    status = "processed"
                tables.append(out)
                for n, b in raw.items():
                    sources.append({"archive": archive.name, "source_layout": source_layout, "archive_bytes": archive.stat().st_size, "member": n, "bytes": len(b), "sha256": sha(b), "crc32": f"{z.getinfo(n).CRC:08x}"})
                print(f"{status}: experiment {exp}, cell {cell}, {len(out)} checkups", flush=True)
    dataset_id = sha("".join(fingerprints).encode())[:16]
    dest = base / f"{VERSION}_{dataset_id}"
    dest.mkdir(parents=True, exist_ok=True)
    d = pd.concat(tables, ignore_index=True).sort_values(["experiment_id", "cell_id", "checkup_idx"]).reset_index(drop=True)
    if d["row_id"].duplicated().any():
        raise ValueError("Duplicate canonical row keys")
    write_table(dest / "kirkaldy_raw_checkups.parquet", d)
    write_table(dest / "kirkaldy_raw_checkups.csv", d)
    mono = d[["row_id", "cell_id", "checkup_idx", "Q_Ah", "Q0_Ah"]].copy()
    mono["Q_cummin_Ah_sensitivity"] = mono.groupby("cell_id")["Q_Ah"].cummin()
    mono["SOH_Q0_cummin_sensitivity"] = mono["Q_cummin_Ah_sensitivity"] / mono["Q0_Ah"]
    mono["changed_by_cummin"] = mono["Q_cummin_Ah_sensitivity"] != mono["Q_Ah"]
    write_table(dest / "kirkaldy_monotonic_sensitivity.parquet", mono)
    write_table(dest / "monotonicity_audit.csv", mono.loc[mono["changed_by_cummin"]])
    write_table(dest / "throughput_discrepancy_audit.csv", d.loc[d["throughput_summary_mismatch"] | d["energy_summary_mismatch"]])
    write_table(dest / "source_member_manifest.csv", pd.DataFrame(sources))
    write_table(dest / "archive_inventory.csv", pd.DataFrame(inventory))
    counts = d.groupby(["experiment_id", "condition_id", "cell_id"], as_index=False).agg(observations=("row_id", "size"), capacity_min_Ah=("Q_Ah", "min"), capacity_max_Ah=("Q_Ah", "max"), start_date=("rpt_date", "min"), end_date=("rpt_date", "max"))
    for k in [3, 5, 10]:
        counts[f"evaluable_K{k}"] = counts["observations"] > k
        counts[f"forecast_points_K{k}"] = (counts["observations"] - k).clip(lower=0)
    write_table(dest / "cell_qc.csv", counts)
    write_table(dest / "missingness.csv", d.isna().sum().rename_axis("field").reset_index(name="missing_rows"))
    population = []
    for exp, group in counts.groupby("experiment_id"):
        for k in [3, 5, 10]:
            eligible = group[group[f"evaluable_K{k}"]]
            population.append({"experiment_id": exp, "K": k, "total_cells": len(group), "evaluable_cells": len(eligible), "evaluable_conditions": eligible["condition_id"].nunique(), "forecast_points": int(group[f"forecast_points_K{k}"].sum())})
    write_table(dest / "context_population.csv", pd.DataFrame(population))
    splits = [{"held_out_experiment": held, "cell_id": row.cell_id, "experiment_id": row.experiment_id, "role": "external_test" if row.experiment_id == held else "adaptation_pool_requires_inner_group_split"} for held in EXPERIMENTS for row in counts.itertuples()]
    write_table(dest / "leave_experiment_out_manifest.csv", pd.DataFrame(splits))
    summary = {"rows": len(d), "cells": d["cell_id"].nunique(), "conditions": d["condition_id"].nunique(), "experiments": d["experiment_id"].nunique(), "duplicate_row_keys": int(d["row_id"].duplicated().sum()), "raw_capacity_increases": int(d["raw_capacity_increase"].sum()), "monotonic_sensitivity_changed_rows": int(mono["changed_by_cummin"].sum()), "throughput_mismatch_rows": int(d["throughput_summary_mismatch"].sum()), "energy_mismatch_rows": int(d["energy_summary_mismatch"].sum()), "capacity_range_Ah": [float(d.Q_Ah.min()), float(d.Q_Ah.max())], "evaluable_cells": {str(k): int(counts[f"evaluable_K{k}"].sum()) for k in [3, 5, 10]}, "status": "sparse_capacity_complete; common_telemetry_exposure_not_frozen"}
    write_json(dest / "qc_summary.json", summary)
    write_json(dest / "preprocessing_manifest.json", {"version": VERSION, "dataset_id": dataset_id, "adapter_sha256": code_sha, "metadata_sha256": meta_sha, "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "checksum_scope": "SHA256 of every consumed compact ZIP member; ZIP CRC verified on read. Whole 49GB archive hashes not recomputed in this run.", "normalization": "Per-cell first RPT0 C/10 capacity only. No fitted scaler or target-test fitting.", "dates": "Naive dates; no timezone or sub-day precision asserted.", "source": "https://zenodo.org/records/10637534", "summary": summary})
    atomic_bytes(dest / "kirkaldy_preprocess.py", Path(__file__).read_bytes())
    atomic_bytes(dest / "experiment_metadata.xlsx", metadata.read_bytes())
    checksums = [{"file": p.name, "sha256": file_sha(p)} for p in sorted(dest.iterdir()) if p.is_file() and p.name not in {"artifact_sha256.csv", "COMPLETE.json"}]
    write_table(dest / "artifact_sha256.csv", pd.DataFrame(checksums))
    write_json(dest / "COMPLETE.json", {"dataset_id": dataset_id, "artifact_manifest_sha256": file_sha(dest / "artifact_sha256.csv")})
    bundle = base / f"{dest.name}.zip"
    temp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    temp.replace(bundle)
    print(json.dumps(summary, indent=2))
    print(f"OUTPUT: {dest}\nBUNDLE: {bundle}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input" if os.name != "nt" and Path("/kaggle/input").exists() else r"F:\battery dataset")
    parser.add_argument("--output-root", default="/kaggle/working/kirkaldy_processed" if os.name != "nt" and Path("/kaggle/working").exists() else str(Path(__file__).resolve().parent / "kirkaldy_processed"))
    args = parser.parse_args()
    run(args.input_root, args.output_root)
