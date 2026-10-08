"""Precision classification and sample label-marker alignment, never model inputs."""
import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from build_luh_interval_features import OFFSET,RAW,OUT,ROOT

events=pd.read_csv(OUT/'EFC_decrease_events.csv')
events['drop_at_source_precision']=(-events.delta_EFC).round(4)
events['above_one_output_unit']=events.drop_at_source_precision>0.0001
events['float32_ulp_before']=np.spacing(events.EFC_before.to_numpy(dtype=np.float32)).astype(float)
events.to_csv(OUT/'EFC_decrease_precision_audit.csv',index=False)
with zipfile.ZipFile(ROOT/'luh_canonical_capacity_v2_3999.zip') as z:
    canonical=pd.read_csv(z.open('luh_canonical_capacity_v2.csv'))
intervals=pd.read_parquet(OUT/'historical_intervals.parquet')
worst=intervals.loc[intervals.anchor_minus_canonical_EFC.idxmin(),'cell_id']
cells=sorted(set(['P001_1_S01_C10','P063_2_S17_C05','P074_2_S18_C03',worst]))
rows=[]
for cell in cells:
    parts=[]
    for chunk in pd.read_csv(RAW/f'cell_log_age_30s_{cell}.csv',sep=';',
        usecols=['timestamp_s','cap_aged_est_Ah','EFC'],chunksize=50000):
        parts.append(chunk[chunk.cap_aged_est_Ah.notna()])
    marks=pd.concat(parts,ignore_index=True)
    for c in canonical[canonical.cell_id==cell].itertuples(index=False):
        differences=marks.timestamp_s+OFFSET-c.timestamp_s
        m=marks.loc[differences.abs().idxmin()]
        rows.append(dict(cell_id=cell,checkup_idx=c.checkup_idx,
            marker_minus_canonical_seconds=float(m.timestamp_s+OFFSET-c.timestamp_s),
            marker_minus_canonical_Q_Ah=float(m.cap_aged_est_Ah-c.Q_Ah),
            marker_minus_canonical_EFC=float(m.EFC-c.EFC_total)))
checks=pd.DataFrame(rows);checks.to_csv(OUT/'sample_capacity_marker_alignment.csv',index=False)
summary=dict(flagged_events=len(events),exactly_one_output_unit=int((events.drop_at_source_precision==.0001).sum()),
    above_one_output_unit=int(events.above_one_output_unit.sum()),largest_drop=float(events.drop_at_source_precision.max()),
    marker_sample_cells=cells,marker_sample_checkups=len(checks),
    markers_within_expected_0_to_30s=int(checks.marker_minus_canonical_seconds.between(0,30).sum()),
    marker_max_abs_Q_difference_Ah=float(checks.marker_minus_canonical_Q_Ah.abs().max()),
    marker_EFC_difference_range=[float(checks.marker_minus_canonical_EFC.min()),float(checks.marker_minus_canonical_EFC.max())],
    interpretation='Tiny output-precision decreases, compatible with float32/resampling effects; original raw counters needed for definitive cause. Marker labels used only for alignment audit.')
(OUT/'alignment_details_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
