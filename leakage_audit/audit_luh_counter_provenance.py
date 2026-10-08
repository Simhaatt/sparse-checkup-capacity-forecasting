"""Bounded provenance tests using owned processed tables and existing F: sources."""
import hashlib, io, json, tarfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'luh_counter_provenance_v1'
AUTHOR = Path(r'F:\battery dataset\reference_code\luh_author_reference')
SOURCE = Path(r'F:\battery datasets\10.35097-1969.tar')


def main():
    OUT.mkdir(exist_ok=True)
    markers = pd.read_csv(ROOT/'luh_interval_features_v1/sample_capacity_marker_alignment.csv')
    with zipfile.ZipFile(ROOT/'luh_canonical_capacity_v2_3999.zip') as z:
        canonical = pd.read_csv(z.open('luh_canonical_capacity_v2.csv'))
        manifest = json.loads(z.read('preprocessing_manifest.json'))
    # Hash the small result archive, never reread the 16.9 GB compact logs.
    sha = hashlib.file_digest(SOURCE.open('rb'), 'sha256').hexdigest()
    assert sha == manifest['source_archive_sha256']
    rows, hashes = [], []
    with tarfile.open(SOURCE) as t:
        member = next(m for m in t if m.name.endswith('/cell_eocv2.zip'))
        with zipfile.ZipFile(io.BytesIO(t.extractfile(member).read())) as z:
            for cell in sorted(markers.cell_id.unique()):
                name = next(n for n in z.namelist() if cell in n and n.endswith('.csv'))
                raw = z.read(name)
                hashes.append(dict(member=name, sha256=hashlib.sha256(raw).hexdigest()))
                e = pd.read_csv(io.BytesIO(raw), sep=';')
                c = canonical[canonical.cell_id == cell].sort_values('checkup_idx')
                # Match the precise canonical measurement, not nearest arbitrary cycle.
                e = e[(e.cyc_condition == 2) & (e.cyc_charged == 0)]
                assert not e.timestamp_s.duplicated().any()
                merged = c.merge(e, on='timestamp_s', suffixes=('', '_source'), validate='one_to_one')
                assert len(merged) == len(c)
                for _, r in merged.iterrows():
                    total = (r.total_q_chg_sum_Ah + r.total_q_dischg_sum_Ah)/6
                    parts = {key: (r[f'total_q_chg_{key}_Ah'] + r[f'total_q_dischg_{key}_Ah'])/6
                             for key in ['CU_RT', 'cyc_OT', 'other_RT', 'other_OT']}
                    rows.append(dict(cell_id=cell, checkup_idx=r.checkup_idx,
                        age_type_code=r.age_type_code, canonical_EFC=r.EFC_total,
                        source_EFC=total, source_Q_Ah=r.cap_aged_est_Ah,
                        canonical_Q_Ah=r.Q_Ah, category_sum_EFC=sum(parts.values()), **parts))
    result = pd.DataFrame(rows).sort_values(['cell_id','checkup_idx'])
    result = result.merge(markers, on=['cell_id','checkup_idx'], validate='one_to_one')
    assert len(result) == len(markers) == 103
    assert np.max(abs(result.source_EFC-result.canonical_EFC)) < 1e-10
    assert np.max(abs(result.source_Q_Ah-result.canonical_Q_Ah)) < 1e-10
    result['compact_marker_EFC'] = result.canonical_EFC + result.marker_minus_canonical_EFC
    result['residual_same_total'] = result.compact_marker_EFC-result.source_EFC
    result['residual_exclude_checkups'] = result.compact_marker_EFC-(result.source_EFC-result.CU_RT)
    result['residual_cycling_only'] = result.compact_marker_EFC-result.cyc_OT
    result['residual_exclude_other'] = result.compact_marker_EFC-(result.CU_RT+result.cyc_OT)
    result['residual_bol_offset_corrected'] = result.residual_same_total-result.groupby('cell_id').residual_same_total.transform('first')
    summary = []
    for cell, d in [('ALL', result), *list(result.groupby('cell_id'))]:
        for name in [c for c in result if c.startswith('residual_')]:
            summary.append(dict(cell_id=cell, hypothesis=name, n=len(d),
                mean_abs_residual_EFC=float(d[name].abs().mean()),
                max_abs_residual_EFC=float(d[name].abs().max()),
                within_0p01_EFC=int((d[name].abs() <= .01).sum())))
    result.to_csv(OUT/'sample_counter_comparison.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT/'hypothesis_residuals.csv', index=False)

    # A source-equivalent isolated gap-fill example, not a measured data finding.
    # Endpoint at 100 s is unavailable at a 50 s forecast cutoff.
    def gap(endpoint):
        timestamps = list(range(0,100,2))
        df = pd.DataFrame({'t': timestamps, 'value': np.nan})
        df.loc[0,'value'] = 0.
        df.loc[len(df)-1,'value'] = endpoint
        df['value'] = df.value.interpolate()
        return df.iloc[1:-1]
    a,b = gap(1.),gap(2.)
    before = a.t < 50
    demo = a.copy();demo['value_if_later_endpoint_changes'] = b.value
    demo['before_cutoff_50s'] = before
    demo.to_csv(OUT/'synthetic_endpoint_dependence.csv', index=False)
    assert (a.loc[before,'value'] != b.loc[before,'value']).all()
    files = ['generate_log_age.py','config_main.py','config_labels.py','README.md']
    source_hashes = [dict(path=str(AUTHOR/f), sha256=hashlib.sha256((AUTHOR/f).read_bytes()).hexdigest()) for f in files]
    # Retain exact notebook producer text as provenance, never execute it.
    nb = json.loads((ROOT/'notebook30e3465872.ipynb').read_text())
    snippets=[]
    for index, cell in enumerate(nb['cells']):
        s = cell.get('source','');s = ''.join(s) if isinstance(s,list) else s
        if 'canonical["charge_throughput_total_Ah"]' in s:
            lines=s.splitlines(); start=next(i for i,l in enumerate(lines) if 'canonical["charge_throughput_total_Ah"]' in l)
            snippets.append(f'Notebook cell index {index}, lines {start+1} onward:\n'+'\n'.join(lines[start:start+55]))
    (OUT/'canonical_producer_excerpt.txt').write_text('\n\n'.join(snippets))
    receipt = dict(result_archive=str(SOURCE), result_archive_sha256=sha,
        sampled_cells=sorted(markers.cell_id.unique().tolist()), sampled_checkups=len(result),
        source_member_hashes=hashes, author_source_hashes=source_hashes,
        marker_audit_sha256=hashlib.sha256((ROOT/'luh_interval_features_v1/sample_capacity_marker_alignment.csv').read_bytes()).hexdigest(),
        max_source_vs_canonical_EFC=float(abs(result.source_EFC-result.canonical_EFC).max()),
        max_source_vs_canonical_Q_Ah=float(abs(result.source_Q_Ah-result.canonical_Q_Ah).max()),
        max_category_sum_vs_total_EFC=float(abs(result.category_sum_EFC-result.source_EFC).max()),
        synthetic_earlier_values_changed_by_future_endpoint=int(before.sum()),
        causal_features_approved=False,
        scope='Four prior audited cells: calendar, cyclic, profile, and worst EFC mismatch. Existing marker audit reused; no compact-log rescan.',
        limitations='Local author code is a reference snapshot, not proven exact release generator. A synthetic witness proves possible endpoint dependence, not actual contaminated-bin count.')
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('hashes')}, indent=2))


if __name__ == '__main__':
    main()
