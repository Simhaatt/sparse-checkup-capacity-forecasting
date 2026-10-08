"""Selectors and split-conformal intervals use disjoint inner roles, never outer truth."""
from study import *
import math
OLD=ROOT.parent/'JES_review_extensions'

def canonical_predictions():
    raw=pd.read_csv(ROOT/'results/final_pred.csv')
    assert raw.job_id.nunique()==375
    keys=['family','information','fold','origin','role','cell_id','row']
    counts=raw.groupby(keys).seed.nunique();assert counts.eq(3).all()
    p=raw.groupby(keys)[['Q_true','Q_pred']].mean().reset_index()
    p['method']=p.family+'__'+p.information
    p.loc[p.information.eq('scheduled'),'method']=p.family
    p=p.drop(columns=['family','information'])
    frames=[p]
    simple=pd.concat([pd.read_csv(OLD/'results/arima_pred.csv'),pd.read_csv(OLD/'results/fair_source_pred.csv'),pd.read_csv(ROOT/'results/chronos_pred.csv')],ignore_index=True)
    simple=simple[['label','origin','cell_id','row','Q_true','Q_pred']].rename(columns={'label':'method'})
    parts=pd.read_csv(ROOT/'audit/cell_partitions.csv')
    for fold in range(1,6):
        membership=parts[parts.fold.eq(fold)].set_index('cell_id').role
        b=simple.copy();b['role']=b.cell_id.map(membership);b=b[b.role.isin(['selection','calibration','test'])];b['fold']=fold;frames.append(b)
    semi=pd.read_csv(ROOT/'results/semiempirical_pred.csv');semi['method']='semiempirical';frames.append(semi[['method','fold','origin','role','cell_id','row','Q_true','Q_pred']])
    p=pd.concat(frames,ignore_index=True);keys=['fold','origin','role','cell_id','row']
    members=['ARIMA_AICc','xgb','large_lstm'];wide=p[p.method.isin(members)].pivot(index=keys,columns='method',values='Q_pred')
    assert set(wide.columns)==set(members) and not wide.isna().any().any()
    avg=wide.mean(axis=1).rename('Q_pred').reset_index();truth=p[p.method.eq('ARIMA_AICc')][keys+['Q_true']]
    avg=avg.merge(truth,on=keys,validate='one_to_one');avg['method']='equal_ARIMA_XGB_largeLSTM';p=pd.concat([p,avg],ignore_index=True)
    assert not p.duplicated(['method']+keys).any()
    q0=data().groupby('cell_id').Q0.first();regime=data().groupby('cell_id').age_type.first()
    p['Q0']=p.cell_id.map(q0);p['regime']=p.cell_id.map(regime);p['step']=p.row-p.origin+1
    return p

def selectors(p):
    choices=[];generated=[]
    pools={
        'selector_scheduled':['compact_lstm','gru','large_lstm','xgb','rf','ARIMA_AICc','sqrt_days_lag1','sqrt_k_lag1','persistence','semiempirical','ChronosBolt_small_zero','equal_ARIMA_XGB_largeLSTM'],
        'selector_no_future_schedule':[f'{family}__{mode}' for family in FAMILIES for mode in ['prefix_fixed','history_only']]+['ARIMA_AICc','sqrt_idx_lag1','persistence','ChronosBolt_small_zero']}
    for name,methods in pools.items():
        for (fold,origin),v in p[p.role.eq('selection')&p.method.isin(methods)].groupby(['fold','origin']):
            if name=='selector_no_future_schedule' and origin!=20:continue
            v=v.copy();v['error']=100*abs(v.Q_pred-v.Q_true)/v.Q_true
            cell=v.groupby(['method','regime','cell_id']).error.mean().reset_index()
            fallback=cell.groupby('method').error.mean().sort_values(kind='mergesort').index[0]
            for regime in [1.,2.,3.]:
                g=cell[cell.regime.eq(regime)];score=g.groupby('method').error.mean().sort_values(kind='mergesort')
                chosen=score.index[0] if len(score) else fallback
                choices.append(dict(selector=name,fold=fold,origin=origin,regime=regime,chosen_method=chosen,selection_cell_count=g.cell_id.nunique(),selection_error=score.iloc[0] if len(score) else np.nan,fallback_used=not len(score)))
                take=p[p.fold.eq(fold)&p.origin.eq(origin)&p.regime.eq(regime)&p.method.eq(chosen)].copy()
                take['selected_member']=chosen;take['method']=name;generated.append(take)
    pd.DataFrame(choices).to_csv(ROOT/'tables/inner_selector_rules.csv',index=False)
    return pd.concat([p,*generated],ignore_index=True)

def conformal(p):
    rows=[];radii=[]
    for (method,fold,origin),g in p.groupby(['method','fold','origin']):
        calibration=g[g.role.eq('calibration')];test=g[g.role.eq('test')]
        assert not(set(calibration.cell_id)&set(test.cell_id))
        for step,t in test.groupby('step'):
            c=calibration[calibration.step.eq(step)];scores=(abs(c.Q_true-c.Q_pred)/c.Q0).to_numpy(float);scores.sort();n=len(scores)
            assert c.cell_id.nunique()==len(c)
            for level in [.8,.9]:
                rank=int(math.ceil((n+1)*level));radius=float(scores[rank-1]) if n and rank<=n else np.inf
                radii.append(dict(method=method,fold=fold,origin=origin,step=step,nominal=level,n_calibration_cells=n,quantile_rank=rank,radius_SOH=radius,finite=bool(np.isfinite(radius))))
                u=t[['method','fold','origin','cell_id','row','step','Q_true','Q_pred','Q0']].copy();u['nominal']=level;u['radius_SOH']=radius
                u['lower']=np.maximum(0,u.Q_pred-radius*u.Q0);u['upper']=u.Q_pred+radius*u.Q0
                u['covered']=u.Q_true.between(u.lower,u.upper);u['width_SOH']=(u.upper-u.lower)/u.Q0;u['finite']=np.isfinite(radius);rows.append(u)
    intervals=pd.concat(rows,ignore_index=True);intervals.to_csv(ROOT/'results/conformal_intervals.csv',index=False)
    pd.DataFrame(radii).to_csv(ROOT/'tables/conformal_radii.csv',index=False)
    summary=[]
    for (method,origin,step,level),g in intervals.groupby(['method','origin','step','nominal']):
        finite=g[g.finite];summary.append(dict(method=method,origin=origin,step=step,nominal=level,n_test_cells=len(g),coverage_all=g.covered.mean(),finite_fraction=g.finite.mean(),coverage_finite=finite.covered.mean(),mean_width_SOH_finite=finite.width_SOH.mean(),mean_width_SOH_all=g.width_SOH.mean()))
    pd.DataFrame(summary).to_csv(ROOT/'tables/conformal_coverage_by_horizon.csv',index=False)
    cell=intervals.groupby(['method','origin','nominal','cell_id']).agg(coverage_all=('covered','mean'),finite_fraction=('finite','mean'),mean_width_SOH_all=('width_SOH','mean')).reset_index()
    cell.groupby(['method','origin','nominal']).agg(n_cells=('cell_id','count'),coverage_equal_cell=('coverage_all','mean'),finite_fraction_equal_cell=('finite_fraction','mean'),mean_width_SOH_all=('mean_width_SOH_all','mean')).reset_index().to_csv(ROOT/'tables/conformal_equal_cell_summary.csv',index=False)

def point_tables(p):
    test=p[p.role.eq('test')].copy();test['APE']=100*abs(test.Q_pred-test.Q_true)/test.Q_true
    cell=test.groupby(['method','origin','cell_id']).APE.mean().reset_index();cell.to_csv(ROOT/'tables/retuned_capacity_cells.csv',index=False)
    cell.groupby(['method','origin']).agg(n_cells=('cell_id','count'),Q_MAPE=('APE','mean'),cell_sd=('APE','std')).reset_index().to_csv(ROOT/'tables/retuned_capacity_summary.csv',index=False)
    test.groupby(['method','origin','step']).agg(n_cells=('cell_id','nunique'),Q_APE=('APE','mean')).reset_index().to_csv(ROOT/'tables/retuned_error_by_horizon.csv',index=False)
    d=data();start={(cid,o):ordered(g).Q.iloc[o-1]/g.Q0.iloc[0] for cid,g in d.groupby('cell_id') if cid in cohort(d) for o in ORIGINS};dec=[]
    for (method,o,cid),g in test.groupby(['method','origin','cell_id']):
        g=g.sort_values('row');t=g.Q_true/g.Q0;x=g.Q_pred/g.Q0;eligible=start[cid,o]>=.8;tc=np.flatnonzero(t.to_numpy()<.8);pc=np.flatnonzero(x.to_numpy()<.8)
        dec.append(dict(method=method,origin=o,cell_id=cid,eligible=eligible,true_event=bool(len(tc)),pred_event=bool(len(pc)),last_APE=g.APE.iloc[-1],bias_pct=(100*(g.Q_pred-g.Q_true)/g.Q_true).mean(),optimistic_share=(g.Q_pred>g.Q_true).mean(),both_cross_error=abs(tc[0]-pc[0]) if len(tc) and len(pc) and eligible else np.nan))
    dec=pd.DataFrame(dec);dec.to_csv(ROOT/'tables/retuned_decision_cells.csv',index=False);out=[]
    for (method,o),g in dec.groupby(['method','origin']):
        e=g[g.eligible];tp=int((e.true_event&e.pred_event).sum());fp=int((~e.true_event&e.pred_event).sum());fn=int((e.true_event&~e.pred_event).sum())
        out.append(dict(method=method,origin=o,n_cells=len(g),n_eligible=len(e),events=tp+fn,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,last_APE=g.last_APE.mean(),bias_pct=g.bias_pct.mean(),optimistic_share=g.optimistic_share.mean(),both_cross_error=e.both_cross_error.mean()))
    pd.DataFrame(out).to_csv(ROOT/'tables/retuned_decision_summary.csv',index=False)

def main():
    p=canonical_predictions();p=selectors(p);p.to_csv(ROOT/'results/canonical_inner_and_test_pred.csv',index=False);conformal(p);point_tables(p)
    parts=pd.read_csv(ROOT/'audit/cell_partitions.csv');checked=[]
    for fold,g in p.groupby('fold'):
        mapping=parts[parts.fold.eq(fold)].set_index('cell_id').role
        checked.append(bool(g.role.eq(g.cell_id.map(mapping)).all()))
    assert all(checked)
    receipt=dict(passed=True,folds=5,main_final_jobs=375,ensemble='Equal weights ARIMA/XGB/large LSTM; three training-seed predictions first averaged per learned predictor.',selector='Regime+prefix-length lookup selected by equal-cell selection-residual MAPE only; fallback to pooled selection winner.',calibration='Calibration cells entirely disjoint from fit, hyperparameter-selection and outer test. Independent calibration scores at each horizon.',levels=[.8,.9],bounds='Symmetric absolute-SOH residual interval; lower clipped at0; rank ceil((n+1)*nominal). Infinite if insufficient finite-sample calibration count.',limitations='Empirical marginal horizon coverage; no simultaneous-trajectory guarantee. Exchangeability is an assumption; retrospective tail availability, repeated test conditions and conditioning on a planned schedule limit deployment interpretation.',partition_sha256=sha(ROOT/'audit/cell_partitions.csv'))
    (ROOT/'audit/inner_analysis_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('INNER_ANALYSIS_PASS',flush=True)
if __name__=='__main__':main()
