"""Audit direct raw EOC capacity provenance on the three previously reconstructed cells."""
from pathlib import Path
import sys,io,ast,json,hashlib,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'JES_review_extensions'
sys.path.insert(0,str(OLD))
from audit_raw_resistance import Ranged

def main():
    code=(OLD/'audit/matched_preprocessing.py').read_text(encoding='utf8')
    tree=ast.parse(code)
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['choose_checkup_key','extract_re_rct_from_eis']]
    import os
    ns={'np':np,'pd':pd,'os':os}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'matched_preprocessing.py','exec'),ns)
    extract=ns['extract_re_rct_from_eis']
    cells=json.loads((OLD/'audit/raw_resistance_receipt.json').read_text())['sample_cells']
    source=pd.read_csv(ROOT/'experiments/inputs/phase1_cv_all_rows.csv')
    dest=ROOT/'audit/raw_capacity_samples';dest.mkdir(exist_ok=True)
    remote=Ranged();summ=[];points=[];files=[]
    with zipfile.ZipFile(remote) as z:
        for cid in cells:
            name=next(n for n in z.namelist() if Path(n).name==f'cell_eocv2_{cid}.csv')
            raw_bytes=z.read(name) # zipfile validates the member CRC
            path=dest/f'cell_eocv2_{cid}.csv';path.write_bytes(raw_bytes)
            files.append(dict(cell_id=cid,member=name,bytes=len(raw_bytes),sha256=hashlib.sha256(raw_bytes).hexdigest()))
            raw=pd.read_csv(io.BytesIO(raw_bytes),sep=';').sort_values('timestamp_s',kind='mergesort').reset_index(drop=True)
            if 'checkup_idx' not in raw:raw['checkup_idx']=np.arange(len(raw),dtype=float)
            raw['checkup_idx']=pd.to_numeric(raw.checkup_idx,errors='coerce').astype(float)
            raw['Q_direct']=pd.to_numeric(raw.cap_aged_est_Ah,errors='coerce')
            eis=extract(str(OLD/'audit/raw_samples'/f'cell_eisv2_{cid}.csv')).rename(columns={'Re':'Re_eis','Rct':'Rct_eis'})
            eis['checkup_idx']=eis.checkup_idx.astype(float)
            merged=pd.merge_asof(raw.sort_values('checkup_idx'),eis.sort_values('checkup_idx'),on='checkup_idx',direction='nearest')
            original=raw.set_index('checkup_idx').Q_direct
            q_unchanged=np.allclose(merged.Q_direct,merged.checkup_idx.map(original),equal_nan=True)
            for c in ['Re_eis','Rct_eis']:merged[c]=merged[c].ffill().bfill()
            filtered=merged[merged.Q_direct.notna()&merged.Re_eis.notna()&merged.Rct_eis.notna()]
            filtered=filtered[filtered.cyc_condition.eq(2)&filtered.cyc_charged.eq(0)]
            filtered=filtered[(filtered.Q_direct>0)&(filtered.Re_eis>0)&(filtered.Rct_eis>0)]
            if 'soh_cap' in filtered:filtered=filtered[filtered.soh_cap.between(0,105)]
            filtered=filtered.sort_values('checkup_idx',kind='mergesort').copy()
            filtered['Q_past_cummin']=filtered.Q_direct.cummin()
            delivered=source[source.cell_id.eq(cid)].sort_values('checkup_idx')
            m=delivered.merge(filtered[['checkup_idx','timestamp_s','Q_direct','Q_past_cummin']],on='checkup_idx',suffixes=('_delivered','_raw'),validate='one_to_one')
            unchanged_prefix=True
            q=filtered.Q_direct.to_numpy(float)
            baseline=np.minimum.accumulate(q)
            for cut in range(1,len(q)):
                altered=q.copy();altered[cut:]=np.linspace(.001,.01,len(q)-cut)
                unchanged_prefix &= np.array_equal(np.minimum.accumulate(altered)[:cut],baseline[:cut])
            err=float(abs(m.Q-m.Q_past_cummin).max())
            passed=bool(q_unchanged and unchanged_prefix and len(m)==len(delivered)==len(filtered) and err<2e-6 and np.allclose(m.timestamp_s_delivered,m.timestamp_s_raw,atol=0,rtol=0) and filtered.timestamp_s.diff().dropna().ge(0).all())
            summ.append(dict(cell_id=cid,raw_eoc_rows=len(raw),protocol_rows=len(filtered),delivered_rows=len(delivered),merge_preserves_left_Q=bool(q_unchanged),max_Q_reconstruction_difference=err,prefix_future_tamper_invariant=bool(unchanged_prefix),timestamp_order_past_only=bool(filtered.timestamp_s.diff().dropna().ge(0).all()),passed=passed))
            m['Q_reconstruction_difference']=m.Q-m.Q_past_cummin;points.append(m[['cell_id','checkup_idx','timestamp_s_delivered','timestamp_s_raw','Q','Q_direct','Q_past_cummin','Q_reconstruction_difference']])
            print('RAW_CAPACITY',summ[-1],flush=True)
    pd.DataFrame(summ).to_csv(ROOT/'tables/raw_capacity_check.csv',index=False)
    pd.concat(points).to_csv(ROOT/'tables/raw_capacity_alignment.csv',index=False)
    pd.DataFrame(files).to_csv(ROOT/'audit/raw_capacity_member_hashes.csv',index=False)
    receipt=dict(passed=all(s['passed'] for s in summ),sample_cells=cells,range_bytes_read=remote.bytes,source_code_sha256=hashlib.sha256(code.encode()).hexdigest(),capacity_column='cap_aged_est_Ah',operations='Direct EOC capacity retained through EIS nearest join; row-local protocol/validity filters; past cumulative minimum. No Q interpolation, nearest right-label assignment or whole-cell Q clipping.',scope='Exact reconstruction on three specified cells plus matched processing-code audit; upstream cap_aged_est_Ah estimation algorithm was not audited. Corpus inclusion/protocol decisions are retrospective.')
    (ROOT/'audit/raw_capacity_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    if not receipt['passed']:raise RuntimeError('Raw capacity reconstruction failed; inspect before training.')
    print('RAW_CAPACITY_CHECK_PASS',flush=True)
if __name__=='__main__':main()
