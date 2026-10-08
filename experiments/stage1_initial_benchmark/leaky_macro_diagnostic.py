"""Historical processed-target macro flattery; aligned future diagnostic is unavailable."""
from statistics_r3 import *

def main():
    metadata('leaky_macro_diagnostic');d=data();rows=[]
    for origin in [20,10]:
        for cid,g0 in d.groupby('cell_id'):
            if cid not in cohort():continue
            g=ordered(g0);q=g.Q.iloc[origin:];r=g.Re.iloc[origin:];qp=g.Q.iloc[origin-1];rp=g.Re.iloc[origin-1]
            Q=float((100*abs(q-qp)/q).mean());R=float((100*abs(r-rp)/r).mean());rows.append(dict(method='persistence',origin=origin,cell_id=cid,Q_MAPE=Q,Re_MAPE=R,joint_macro_MAPE=.5*(Q+R),Re_tail_range=float(r.max()-r.min()),label_stream='historical_processed_export'))
    ar=pd.read_csv(ROOT/'results/reused_arima_pred.csv')
    for (origin,cid),g in ar[ar.origin.isin([20,10])].groupby(['origin','cell_id']):
        Q=float((100*abs(g.Q_true-g.Q_pred)/g.Q_true).mean());R=float((100*abs(g.Re_true-g.Re_pred)/g.Re_true).mean());rows.append(dict(method='ARIMA_AICc',origin=origin,cell_id=cid,Q_MAPE=Q,Re_MAPE=R,joint_macro_MAPE=.5*(Q+R),Re_tail_range=float(g.Re_true.max()-g.Re_true.min()),label_stream='historical_processed_export'))
    cell=pd.DataFrame(rows);cell.to_csv(ROOT/'tables/historical_macro_cells.csv',index=False);summary=[]
    for (method,origin),g in cell.groupby(['method','origin']):
        for target in ['Q_MAPE','Re_MAPE','joint_macro_MAPE']:
            lo,hi=bootstrap_ci(g[target]);summary.append(dict(method=method,origin=origin,target=target,n_cells=len(g),mean=g[target].mean(),ci95_low=lo,ci95_high=hi,status='diagnostic_only_invalid_resistance_labels'))
    pd.DataFrame(summary).to_csv(ROOT/'tables/historical_macro_summary.csv',index=False)
    aligned=pd.read_csv(ROOT/'results/aligned_re_v2.csv');g=d.sort_values(['cell_id','timestamp_s']).copy();g['row']=g.groupby('cell_id').cumcount();a=g.merge(aligned[['cell_id','checkup_idx','Re_mask']],on=['cell_id','checkup_idx'],validate='one_to_one')
    unavailable=[]
    for origin in [20,10]:
        count=int(a[a.cell_id.isin(cohort())&a.row.ge(origin)].Re_mask.sum());assert count==0
        unavailable.append(dict(origin=origin,aligned_future_Re_labels=count,aligned_macro_MAPE=np.nan,status='unavailable_by_explicit_user_decision',reason='Fixedbackward14dayjoin yieldsno alignedfutureRe labels; toleranceunchanged, noforwardjoin or fabricatedlabel.'))
    pd.DataFrame(unavailable).to_csv(ROOT/'tables/aligned_macro_unavailable.csv',index=False)
    (ROOT/'audit/macro_diagnostic_receipt.json').write_text(json.dumps(dict(passed=True,old_processed_diagnostic=True,aligned_future_labels=0,aligned_comparison='Unavailablebyexplicithumaninstruction',interpretation='Jointmacro canappearbetter simplybecause fakeRe isflat/easy; it isnot physical resistanceforecasting accuracy. Do notmix intoactiveQranking.'),indent=2),encoding='utf8');print('MACRO_DIAGNOSTIC_COMPLETE')
if __name__=='__main__':main()
