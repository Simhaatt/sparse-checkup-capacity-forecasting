"""Timestamp-aligned raw EIS rebuild; capacity labels are never changed."""
from pathlib import Path
import sys,os,json,ast,zipfile,zlib,hashlib,io
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent
sys.path.insert(0,str(OLD))
from audit_raw_resistance import Ranged

def sweeps(path):
    raw=pd.read_csv(path,sep=';');d=raw.copy()
    for f,v in [('is_rt',1),('soc_nom',50),('valid',1)]:
        if f not in d:raise ValueError('Required raw RT/SOC/validity field missing: '+f)
        d=d[pd.to_numeric(d[f],errors='coerce').eq(v)]
    rows=[]
    if not len(d):return pd.DataFrame(columns=['eis_timestamp_s','Re_aligned_sweep','sequence'])
    if 'seq_nr' not in d:raise ValueError('No raw sweep sequence identifier')
    for key,g in d.groupby('seq_nr'):
        # Strict RT/SOC50 grouping uses the actual sequence, never a made-up checkup ordinal.
        g=g.sort_values('freq_Hz',ascending=False);finite=np.isfinite(pd.to_numeric(g.z_re_comp_mOhm,errors='coerce'))
        g=g[finite]
        if len(g)<3 or g.freq_Hz.nunique()<3:continue
        timestamp=float(pd.to_numeric(g.timestamp_s).max())
        rows.append(dict(eis_timestamp_s=timestamp,Re_aligned_sweep=float(g.z_re_comp_mOhm.iloc[:3].mean()),sequence=str(key),points=len(g),timestamp_start_s=float(g.timestamp_s.min())))
    return pd.DataFrame(rows).sort_values(['eis_timestamp_s','sequence']).drop_duplicates('eis_timestamp_s',keep='last') if rows else pd.DataFrame(columns=['eis_timestamp_s','Re_aligned_sweep','sequence'])

def main():
    source=pd.read_csv(ROOT/'inputs/phase1_cv_all_rows.csv');dest=ROOT/'audit/raw_eis';dest.mkdir(exist_ok=True)
    remote=Ranged();files=[];errors=[]
    with zipfile.ZipFile(remote) as archive:
        membermap={Path(n).name:n for n in archive.namelist()}
        for cid in sorted(source.cell_id.unique()):
            filename=f'cell_eisv2_{cid}.csv';path=dest/filename;old=OLD/'audit/raw_samples'/filename
            try:
                if filename not in membermap:raise FileNotFoundError('Raw EIS member absent')
                member=membermap[filename];info=archive.getinfo(member)
                if not path.exists() and old.exists():path.write_bytes(old.read_bytes())
                if path.exists() and (zlib.crc32(path.read_bytes())&0xffffffff)==info.CRC:
                    acquisition='verified_cached_raw_member'
                else:
                    b=archive.read(member);path.write_bytes(b);acquisition='bounded_remote_zip_member'
                assert (zlib.crc32(path.read_bytes())&0xffffffff)==info.CRC
                files.append(dict(cell_id=cid,file=filename,member=member,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crc32=info.CRC,acquisition=acquisition))
                print('RAW_EIS_READY',len(files),'/',source.cell_id.nunique(),cid,flush=True)
            except Exception as e:
                errors.append(dict(cell_id=cid,error=str(e)));print('RAW_EIS_ERROR',cid,str(e),flush=True)
    pd.DataFrame(files).to_csv(ROOT/'audit/raw_eis_member_manifest.csv',index=False)
    pd.DataFrame(errors,columns=['cell_id','error']).to_csv(ROOT/'audit/raw_eis_acquisition_errors.csv',index=False)
    rows=[];coverage=[];all_sweeps=[]
    for cid,g0 in source.groupby('cell_id'):
        g=g0.sort_values('timestamp_s').copy();path=dest/f'cell_eisv2_{cid}.csv'
        if not path.exists():
            g['Re_aligned']=np.nan;g['eis_timestamp_s']=np.nan;g['Re_mask']=0
        else:
            s=sweeps(path);s['cell_id']=cid;all_sweeps.append(s)
            if len(s):
                joined=pd.merge_asof(g,s,left_on='timestamp_s',right_on='eis_timestamp_s',direction='backward',tolerance=14*86400,suffixes=('','_eis'))
                g=joined;g['Re_aligned']=g.Re_aligned_sweep;g['Re_mask']=g.Re_aligned.notna().astype(int)
            else:g['Re_aligned']=np.nan;g['eis_timestamp_s']=np.nan;g['Re_mask']=0
        g['lag_days']=(g.timestamp_s-g.eis_timestamp_s)/86400
        # Re0 is exposed only when the first aligned observation has actually arrived.
        first=np.flatnonzero(g.Re_mask.to_numpy())
        g['Re0_available']=0.;g['Re0_mask']=0
        if len(first):
            start=int(first[0]);g.iloc[start:,g.columns.get_loc('Re0_available')]=g.Re_aligned.iloc[start];g.iloc[start:,g.columns.get_loc('Re0_mask')]=1
        rows.append(g[['cell_id','timestamp_s','checkup_idx','Re_aligned','Re_mask','Re0_available','Re0_mask','eis_timestamp_s','lag_days']])
        coverage.append(dict(cell_id=cid,age_type=int(g.age_type.iloc[0]),capacity_rows=len(g),raw_file_obtained=path.exists(),aligned_rows=int(g.Re_mask.sum()),aligned_fraction=g.Re_mask.mean(),full_aligned_coverage=bool(g.Re_mask.eq(1).all()),prefix10_aligned=int(g.Re_mask.iloc[:10].sum()),prefix20_aligned=int(g.Re_mask.iloc[:20].sum())))
    aligned=pd.concat(rows,ignore_index=True);pd.DataFrame(coverage).to_csv(ROOT/'tables/aligned_re_coverage.csv',index=False)
    aligned.to_csv(ROOT/'results/aligned_re.csv',index=False);pd.concat(all_sweeps,ignore_index=True).to_csv(ROOT/'results/raw_eis_sweeps.csv',index=False)
    valid=aligned[aligned.Re_mask.eq(1)];assert valid.lag_days.between(0,14).all()
    assert aligned[aligned.Re_mask.eq(0)].Re_aligned.isna().all()
    receipt=dict(passed=True,capacity_cells=source.cell_id.nunique(),raw_cells_obtained=len(files),raw_cells_unavailable=len(errors),capacity_rows=len(aligned),aligned_rows=int(aligned.Re_mask.sum()),range_bytes_read=remote.bytes,join='Backward timestamp as-of; latest completed RT/SOC50 valid raw sweep at or before capacity timestamp;14day tolerance.',sweep_Re='Mean highest3 frequency compensated real-impedance samples; raw sequence identifier grouping; end timestamp=max point timestamp. UnitsmOhm.',forbidden_operations='No imputation into missing Re labels, no cumulative maximum, no whole-cell clipping, no nearest ordinal join.',Re0='First aligned sweep, unavailable before its arrival. Re0 feature must also be frozen at the forecast origin.',scope='All source cells with obtainable EIS; file availability and per-checkup alignment coverage are separately recorded.')
    (ROOT/'audit/aligned_re_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('ALIGNED_RESISTANCE_COMPLETE',flush=True)
if __name__=='__main__':main()
