"""Meaningful adapter and notebook checks against official local source members."""
import contextlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
import nbformat
import pandas as pd
import kirkaldy_preprocess as adapter

ROOT = Path(__file__).resolve().parent
RAW = Path(r"F:\battery dataset")


def expect_failure(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Invalid source was accepted")


meta = pd.read_excel(RAW / "experiment_metadata.xlsx", sheet_name="expt 5").iloc[0]
with zipfile.ZipFile(RAW / (meta["Expt long name"] + ".zip")) as z:
    p = adapter.read_csv(z.read(adapter.member(z, "/Performance Summary/Expt 5 - cell A (10degC) - Processed Data.csv")))
    s = adapter.read_csv(z.read(adapter.member(z, "/Summary per Set/expt 5 - cell A - set_data.csv")))
d = adapter.convert(p, s, meta, {})
assert str(d.loc[1, "rpt_date"].date()) == "2020-08-10"
assert d.loc[1, "elapsed_days"] == 40  # not source degradation days
assert d.loc[0, "SOH_Q0"] == 1
assert d.delta_EFC.isna().all() and d.charge_C.isna().all()
future = p.copy()
future.loc[10:, "C/10 Capacity [mA h]"] *= .8
changed = adapter.convert(future, s, meta, {})
pd.testing.assert_series_equal(d.loc[:9, "SOH_Q0"], changed.loc[:9, "SOH_Q0"])
expect_failure(lambda: adapter.convert(pd.concat([p, p.iloc[[0]]]), s, meta, {}))
expect_failure(lambda: adapter.convert(p.iloc[1:], s, meta, {}))
bad = p.copy()
bad.loc[3, "RPT start date"] = "garbage"
expect_failure(lambda: adapter.convert(bad, s, meta, {}))
bad = p.copy()
bad.loc[2, "C/10 Capacity [mA h]"] = -1
expect_failure(lambda: adapter.convert(bad, s, meta, {}))

nb = nbformat.read(ROOT / "Kirkaldy_Preprocessing_v1.ipynb", as_version=4)
nbformat.validate(nb)
with tempfile.TemporaryDirectory(prefix="kirkaldy_verify_", dir=ROOT) as tmp:
    tmp = Path(tmp)
    with zipfile.ZipFile(ROOT / "kirkaldy_compact_sources.zip") as z:
        # Package was built by this project and all entries must remain within this temp directory.
        for n in z.namelist():
            target = (tmp / "input" / n).resolve()
            assert target.is_relative_to((tmp / "input").resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
    namespace = {"__name__": "__notebook__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code":
                exec(compile(c.source, f"notebook_cell_{i}", "exec"), namespace)
                if "INPUT_ROOT =" in c.source:
                    namespace.update(INPUT_ROOT=tmp / "input", WORK=tmp / "work", OUTPUT_ROOT=tmp / "output")
                    namespace["WORK"].mkdir()
    result = namespace["result_dir"]
    frame = pd.read_parquet(result / "kirkaldy_raw_checkups.parquet")
    assert len(frame) == 511 and frame.cell_id.nunique() == 40
    assert frame.raw_capacity_increase.sum() == 12
    assert frame.loc[frame.experiment_id == "4", "soc_min_scenario"].isna().all()
    canonical_hash = adapter.file_sha(result / "kirkaldy_raw_checkups.parquet")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result2 = namespace["adapter"].run(tmp / "input", tmp / "output")
    assert captured.getvalue().count("resumed:") == 40
    assert canonical_hash == adapter.file_sha(result2 / "kirkaldy_raw_checkups.parquet")
    # A damaged cache must be rebuilt rather than silently accepted.
    part = next((tmp / "output" / "cell_cache").glob("*.parquet"))
    part.write_bytes(b"corrupted test cache")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        namespace["adapter"].run(tmp / "input", tmp / "output")
    assert captured.getvalue().count("processed:") == 1
    assert canonical_hash == adapter.file_sha(result2 / "kirkaldy_raw_checkups.parquet")
    # Independent source-to-target reconciliation for every row in the compact input.
    for archive in (tmp / "input").glob("Expt*.zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if "/Performance Summary/" not in member:
                    continue
                raw = pd.read_csv(z.open(member))
                got = frame[frame.source_member == member].sort_values("source_record_idx")
                assert len(got) == len(raw)
                assert ((got.Q_Ah.to_numpy() - raw['C/10 Capacity [mA h]'].to_numpy() / 1000) == 0).all()
    # Reproduce Kaggle recursively extracting the inner ZIPs, with no ZIP inputs remaining.
    extracted = tmp / 'extracted' / 'datasets' / 'owner' / 'arbitrary-slug'
    extracted.mkdir(parents=True)
    (extracted / 'experiment_metadata.xlsx').write_bytes((tmp / 'input' / 'experiment_metadata.xlsx').read_bytes())
    for archive in (tmp / 'input').glob('Expt*.zip'):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (extracted / name).resolve()
                assert target.is_relative_to(extracted.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    with contextlib.redirect_stdout(io.StringIO()):
        folder_result = namespace['adapter'].run(tmp / 'extracted', tmp / 'folder_output')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(folder_result / 'kirkaldy_raw_checkups.parquet'))
    folder_sources = pd.read_csv(folder_result / 'source_member_manifest.csv')
    assert folder_sources.source_layout.eq('reconstructed_from_extracted_summaries').all()
    zip_sources = pd.read_csv(result / 'source_member_manifest.csv')
    pd.testing.assert_frame_equal(folder_sources[['member','sha256']], zip_sources[['member','sha256']])
    # Duplicate extracted input is rejected instead of selecting an arbitrary copy.
    extra = tmp / 'extracted' / 'duplicate' / 'Performance Summary'
    extra.mkdir(parents=True)
    anchor = next(extracted.rglob('Expt 1 - cell A (10degC) - Processed Data.csv'))
    (extra / anchor.name).write_bytes(anchor.read_bytes())
    expect_failure(lambda: namespace['adapter'].resolve_experiment(tmp / 'extracted', 'Expt 1 - Si-based Degradation', '1', tmp / 'ambiguous_output'))
print('PASS: Kaggle extracted folders produce identical canonical rows and all 120 member SHA256 hashes; duplicate folders rejected.')
print("PASS: notebook schema and all code cells; 511-row source reconciliation; date semantics; no future reference leakage; fail-fast invalid input; 40-cell resume; corrupt cache recovery.")
