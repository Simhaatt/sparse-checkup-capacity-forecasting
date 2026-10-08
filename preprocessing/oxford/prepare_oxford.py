"""Extract measured diagnostic discharge capacity, preserving original checkups/no cummin."""
from study import *
from scipy.io import loadmat
import re,gc

def main():
    path=ROOT/'external/oxford/Oxford_Battery_Degradation_Dataset_1.mat';rows=[];audit=[];duplicates=[]
    for c in range(1,9):
        name=f'Cell{c}';s=loadmat(path,variable_names=[name],simplify_cells=True)[name];records=[]
        for checkup in sorted(s,key=lambda n:int(re.search(r'\d+',n).group())):
            g=s[checkup]
            if 'C1dc' not in g:raise ValueError(f'{name} {checkup}: no diagnostic1C discharge')
            x=g['C1dc'];q=np.asarray(x['q'],float).ravel();t=np.asarray(x['t'],float).ravel();v=np.asarray(x['v'],float).ravel()
            assert len(q)==len(t)==len(v)>1 and np.isfinite(q).all() and np.isfinite(t).all()
            capacity=abs(q[-1]-q[0])/1000
            assert 0<capacity<1.5 and t[-1]>t[0] and v[0]>v[-1]
            records.append(dict(cell_id=name,cycle=int(re.search(r'\d+',checkup).group()),Q=capacity,serial_time_end=t[-1],diagnostic_points=len(q),voltage_start=v[0],voltage_end=v[-1],mean_diagnostic_temperature=float(np.mean(x['T'])),charge_monotone=bool(np.diff(q).max()<1e-5),capacity_range_difference=abs(abs(q[-1]-q[0])-(q.max()-q.min()))/1000,diagnostic_sha256=hashlib.sha256(q.tobytes()+t.tobytes()+v.tobytes()).hexdigest()))
        g=pd.DataFrame(records).sort_values('cycle').reset_index(drop=True)
        duplicate=g.diagnostic_sha256.duplicated(keep='first');duplicates.append(g[duplicate]);n_duplicate=int(duplicate.sum())
        g=g[~duplicate].reset_index(drop=True);g['checkup_idx']=np.arange(len(g));g['Q0']=g.Q.iloc[0]
        g['Q_SOH']=g.Q/g.Q0;g['k_exp']=g.cycle-g.cycle.iloc[0]
        # Values near735,000 with a1h diagnostic increment near1/24 are MATLAB serial days,
        # despite the README's generic "seconds" label. Only relative elapsed time is derived.
        g['timestamp_s']=(g.serial_time_end-g.serial_time_end.iloc[0])*86400
        g['t_days']=g.timestamp_s/86400;g['temperature']=40.;g['age_type']=3
        assert g.timestamp_s.diff().dropna().gt(0).all() and len(g)>20
        rows.append(g);audit.append(dict(cell_id=name,checkups=len(g),removed_exact_duplicate_checkups=n_duplicate,capacity_Ah_initial=g.Q.iloc[0],capacity_Ah_last=g.Q.iloc[-1],tail_after20=len(g)-20,raw_capacity_increases=int(g.Q.diff().gt(0).sum()),max_capacity_endpoint_range_difference=g.capacity_range_difference.max()))
        print('OXFORD_CELL',audit[-1],flush=True);del s;gc.collect()
    pd.concat(rows,ignore_index=True).to_csv(ROOT/'experiments/inputs/oxford_capacity_checkups.csv',index=False)
    pd.DataFrame(audit).to_csv(ROOT/'tables/oxford_input_audit.csv',index=False)
    pd.concat(duplicates,ignore_index=True).to_csv(ROOT/'audit/oxford_exact_duplicate_diagnostics.csv',index=False)
    receipt=dict(passed=True,cells=8,raw_mat_sha256=sha(path),download_receipt_sha256=sha(ROOT/'audit/oxford_download_receipt.json'),capacity='Absolute diagnostic C1dc charge endpoint difference in mAh divided by1000; no smoothing/cummin/future matching.',time='Observed MATLAB serial days converted to relative seconds; diagnostic endpoint is availability timestamp.',checkup_protocol='1C diagnostic tests every100 drive cycles;40C chamber;2C charging and Urban Artemis discharging according to original README.',source_chemistry='Commercial Kokam SLPB533459H4 pouch cells; precise chemistry not asserted from this dataset README alone.',planned_evaluation='Eight leave-one-cell-out outer tests; next cyclically numbered cell is inner validation; other six are training. Five learned families on Q/Q0 history only plus ARIMA/sqrt/persistence/Chronos controls. Structures selected on main-source inner cells are frozen before Oxford test scoring.')
    (ROOT/'audit/oxford_input_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('OXFORD_INPUT_PASS',flush=True)
if __name__=='__main__':main()
