"""Native central80% quantile coverage; do not fabricate native90% tails."""
from statistics_r3 import *

def main():
    metadata('native_chronos_coverage');rows=[];cells=[];source=data();q0=source.groupby('cell_id').Q0.first();age=source.groupby('cell_id').age_type.first()
    for file,method in [('reused_chronos_pred.csv','ChronosBolt_small_zero'),('chronos_tiny_pred.csv','ChronosBolt_tiny_zero')]:
        p=pd.read_csv(ROOT/'results'/file);p['method']=method;p['lower']=np.maximum(0,p.native_p10);p['upper']=p.native_p90;p['age_type']=p.cell_id.map(age)
        assert p.upper.ge(p.lower).all() and np.isfinite(p[['lower','upper']]).all().all()
        p['covered']=p.Q_true.between(p.lower,p.upper);p['width_Ah']=p.upper-p.lower;p['width_SOH']=p.width_Ah/p.cell_id.map(q0)
        c=p.groupby(['method','origin','age_type','cell_id']).agg(coverage=('covered','mean'),width_Ah=('width_Ah','mean'),width_SOH=('width_SOH','mean')).reset_index();cells.append(c)
        for origin,g in c.groupby('origin'):
            lo,hi=bootstrap_ci(g.coverage);rows.append(dict(method=method,origin=origin,nominal=.8,n_cells=len(g),coverage_equal_cell=g.coverage.mean(),coverage_ci95_low=lo,coverage_ci95_high=hi,mean_width_Ah=g.width_Ah.mean(),mean_width_SOH=g.width_SOH.mean()))
    pd.concat(cells,ignore_index=True).to_csv(ROOT/'tables/native_chronos_coverage_cells.csv',index=False);pd.DataFrame(rows).to_csv(ROOT/'tables/native_chronos_coverage.csv',index=False)
    pd.concat(cells,ignore_index=True).groupby(['method','origin','age_type']).agg(n_cells=('cell_id','count'),coverage=('coverage','mean'),width_Ah=('width_Ah','mean'),width_SOH=('width_SOH','mean')).reset_index().to_csv(ROOT/'tables/native_chronos_by_type.csv',index=False)
    (ROOT/'audit/native_chronos_coverage_receipt.json').write_text(json.dumps(dict(passed=True,models=['ChronosBolt_small_zero','ChronosBolt_tiny_zero'],origins=[10,15,20],source_cells=104,native_level=.8,native90='Notreported: trainedquantiles0.1..0.9 do notprovide0.05/0.95 tails; separateconformal90%computedelsewhere.',interpretation='Empirical native quantilecoverage, not a coverageguarantee. Order-index forecasts ignoreirregular elapsed spacing.'),indent=2),encoding='utf8');print('NATIVE_CHRONOS_COVERAGE_COMPLETE')
if __name__=='__main__':main()
