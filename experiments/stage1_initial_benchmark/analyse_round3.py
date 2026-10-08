"""Final point/combination/selector/interval evidence using the declared fold roles."""
from statistics_r3 import *
import math

def combinations(p):
    selected=best_neural_table();copies=[]
    for r in selected.itertuples():
        q=p[p.fold.eq(r.fold)&p.origin.eq(r.origin)&p.method.eq(r.best_neural)].copy();q['selected_member']=r.best_neural;q['method']='best_neural';q['family']='best_neural';copies.append(q)
    p=pd.concat([p,*copies],ignore_index=True);keys=['fold','origin','role','seed','cell_id','row'];members=['ARIMA_AICc','xgb','best_neural']
    wide=p[p.method.isin(members)].pivot(index=keys,columns='method',values='Q_pred');assert not wide.isna().any().any()
    meta=p[p.method.eq('ARIMA_AICc')][keys+['Q_true','Q0','age_type','step']]
    recipes={'avg_ARIMA_XGB':['ARIMA_AICc','xgb'],'avg_ARIMA_NN':['ARIMA_AICc','best_neural'],'avg_XGB_NN':['xgb','best_neural'],'avg_ARIMA_XGB_NN':members};generated=[]
    for name,components in recipes.items():
        q=wide[components].mean(axis=1).rename('Q_pred').reset_index().merge(meta,on=keys,validate='one_to_one');q['method']=name;q['family']='combination';q['variant']='equal_mean';generated.append(q)
    return pd.concat([p,*generated],ignore_index=True)

def selector(p):
    pool=FAMILIES+['ARIMA_AICc','sqrt_days_lag1','sqrt_k_lag1','sqrt_idx_lag1','persistence','ChronosBolt_small_zero','semiempirical','avg_ARIMA_XGB','avg_ARIMA_NN','avg_XGB_NN','avg_ARIMA_XGB_NN']
    if 'ChronosBolt_tiny_zero' in set(p.method):pool.append('ChronosBolt_tiny_zero')
    rules=[];generated=[]
    for (fold,origin),v in p[p.role.eq('validation')&p.method.isin(pool)].groupby(['fold','origin']):
        c=cell_errors(v);ages=data().groupby('cell_id').age_type.first();c['age_type']=c.cell_id.map(ages);overall=c.groupby('method').APE.mean().sort_values(kind='mergesort');fallback=overall.index[0]
        for age in [1,2,3]:
            a=c[c.age_type.eq(age)];score=a.groupby('method').APE.mean().sort_values(kind='mergesort');chosen=score.index[0] if len(score) else fallback
            rules.append(dict(fold=fold,origin=origin,age_type=age,method=chosen,validation_cells=a.cell_id.nunique(),inner_error=score.iloc[0] if len(score) else overall.iloc[0],fallback=not len(score)))
            q=p[p.fold.eq(fold)&p.origin.eq(origin)&p.age_type.eq(age)&p.method.eq(chosen)].copy();q['selected_member']=chosen;q['method']='regime_origin_selector';q['family']='selector';q['variant']='inner_lookup';generated.append(q)
    pd.DataFrame(rules).to_csv(ROOT/'tables/selector_rules.csv',index=False)
    return pd.concat([p,*generated],ignore_index=True)

def point_analysis(p):
    test=p[p.role.eq('test')];cell=cell_errors(test);cell.to_csv(ROOT/'tables/capacity_cell_errors.csv',index=False)
    summaries=summarize(cell,'round3_capacity');summaries.to_csv(ROOT/'tables/capacity_summary.csv',index=False)
    age=data().groupby('cell_id').age_type.first();typed=cell.copy();typed['age_type']=typed.cell_id.map(age)
    typed.groupby(['method','origin','age_type']).agg(n_cells=('cell_id','count'),Q_MAPE=('APE','mean'),cell_sd=('APE','std')).reset_index().to_csv(ROOT/'tables/capacity_by_type.csv',index=False)
    tests=[]
    for origin in [20,10,15]:tests.append(paired(cell,[(origin,m,'ARIMA_AICc') for m in FAMILIES],f'tuned_families_vs_ARIMA_origin{origin}'))
    ablations=[(20,f'{family}__{variant}',family) for family in FAMILIES for variant in ['no_k','static','history_only']]
    tests.append(paired(cell,ablations,'schedule_ablations_vs_full_origin20_all15'))
    for origin in [20,10]:
        comparisons=[(origin,f'{family}__{arm}',family) for family in FAMILIES for arm in ['B','C']]
        tests.append(paired(cell,comparisons,f'resistance_arms_vs_A_origin{origin}_all10'))
        for age_type,g in typed[typed.origin.eq(origin)].groupby('age_type'):
            tests.append(paired(g,comparisons,f'resistance_arms_vs_A_origin{origin}_age{age_type}_all10'))
    ensembles=['avg_ARIMA_XGB','avg_ARIMA_NN','avg_XGB_NN','avg_ARIMA_XGB_NN'];references=['ARIMA_AICc','xgb','best_neural']
    for origin in [20,10,15]:
        tests.append(paired(cell,[(origin,m,ref) for m in ensembles for ref in references],f'equal_combinations_vs_members_origin{origin}_all12'))
        tests.append(paired(cell,[(origin,'regime_origin_selector',ref) for ref in FAMILIES+['ARIMA_AICc','avg_ARIMA_XGB_NN']],f'selector_vs_references_origin{origin}_all7'))
    alltests=pd.concat(tests,ignore_index=True);alltests.to_csv(ROOT/'tables/paired_tests.csv',index=False)
    alltests.groupby('family').agg(comparisons=('comparison','count')).reset_index().to_csv(ROOT/'tables/HOLM_FAMILIES.csv',index=False)
    ranks=[]
    for origin in [20,10]:
        for arm in ['A','B','C']:
            names=FAMILIES if arm=='A' else [f'{f}__{arm}' for f in FAMILIES];g=summaries[summaries.origin.eq(origin)&summaries.method.isin(names)].sort_values('Q_MAPE_equal_cell')
            for rank,row in enumerate(g.itertuples(),1):ranks.append(dict(origin=origin,arm=arm,family=row.method.split('__')[0],rank=rank,Q_MAPE=row.Q_MAPE_equal_cell))
    pd.DataFrame(ranks).to_csv(ROOT/'tables/resistance_arm_ranks.csv',index=False)
    return summaries

def conformal(p):
    radii=[];rows=[]
    for (method,fold,origin,seed),g in p.groupby(['method','fold','origin','seed']):
        c=g[g.role.eq('calibration')];t=g[g.role.eq('test')];assert not(set(c.cell_id)&set(t.cell_id))
        for step,test in t.groupby('step'):
            cal=c[c.step.eq(step)];assert not cal.cell_id.duplicated().any();score=np.sort(abs(cal.Q_true-cal.Q_pred).to_numpy());n=len(score)
            for nominal in [.8,.9]:
                rank=math.ceil((n+1)*nominal);radius=float(score[rank-1]) if n and rank<=n else np.inf
                radii.append(dict(method=method,fold=fold,origin=origin,seed=seed,step=step,nominal=nominal,n_calibration=n,quantile_rank=rank,radius_Ah=radius,finite=np.isfinite(radius)))
                q=test[['method','fold','origin','seed','cell_id','row','step','age_type','Q_true','Q_pred','Q0']].copy();q['nominal']=nominal;q['lower']=np.maximum(0,q.Q_pred-radius);q['upper']=q.Q_pred+radius;q['covered']=q.Q_true.between(q.lower,q.upper);q['width_Ah']=q.upper-q.lower;q['width_SOH']=q.width_Ah/q.Q0;q['finite']=np.isfinite(radius);rows.append(q)
    pd.DataFrame(radii).to_csv(ROOT/'tables/conformal_radii.csv',index=False);I=pd.concat(rows,ignore_index=True);I.to_csv(ROOT/'results/conformal_intervals.csv',index=False)
    # Seed-average within each cell/horizon before averaging cells, also by ageing type.
    cell=I.groupby(['method','origin','nominal','step','age_type','cell_id']).agg(coverage=('covered','mean'),width_Ah=('width_Ah','mean'),width_SOH=('width_SOH','mean'),finite_fraction=('finite','mean')).reset_index()
    for label,keys in [('overall',['method','origin','nominal','step']),('by_type',['method','origin','nominal','step','age_type'])]:
        cell.groupby(keys).agg(n_cells=('cell_id','count'),coverage=('coverage','mean'),mean_width_Ah=('width_Ah','mean'),mean_width_SOH=('width_SOH','mean'),finite_fraction=('finite_fraction','mean')).reset_index().to_csv(ROOT/'tables'/f'conformal_horizon_{label}.csv',index=False)
    c=cell.groupby(['method','origin','nominal','age_type','cell_id'])[['coverage','width_Ah','width_SOH','finite_fraction']].mean().reset_index()
    c.groupby(['method','origin','nominal']).agg(n_cells=('cell_id','count'),coverage_equal_cell=('coverage','mean'),mean_width_Ah=('width_Ah','mean'),mean_width_SOH=('width_SOH','mean'),finite_fraction=('finite_fraction','mean')).reset_index().to_csv(ROOT/'tables/conformal_overall.csv',index=False)
    c.groupby(['method','origin','nominal','age_type']).agg(n_cells=('cell_id','count'),coverage_equal_cell=('coverage','mean'),mean_width_Ah=('width_Ah','mean'),mean_width_SOH=('width_SOH','mean'),finite_fraction=('finite_fraction','mean')).reset_index().to_csv(ROOT/'tables/conformal_by_type.csv',index=False)

def decisions(p):
    d=data();prefix={(cid,o):ordered(g).Q.iloc[o-1]/g.Q0.iloc[0] for cid,g in d.groupby('cell_id') if cid in cohort() for o in [20,10,15]};rows=[]
    for (method,origin,seed,cid),g in p[p.role.eq('test')].groupby(['method','origin','seed','cell_id']):
        g=g.sort_values('row');true=g.Q_true.to_numpy()/g.Q0.iloc[0];pred=g.Q_pred.to_numpy()/g.Q0.iloc[0];tc=np.flatnonzero(true<.8);pc=np.flatnonzero(pred<.8);eligible=prefix[cid,origin]>=.8;h=len(g)
        rows.append(dict(method=method,origin=origin,seed=seed,cell_id=cid,eligible=eligible,true_event=bool(len(tc)),pred_event=bool(len(pc)),true_cross_step=int(tc[0]+1) if len(tc) else h+1,pred_cross_step=int(pc[0]+1) if len(pc) else h+1,last_APE=100*abs(g.Q_pred.iloc[-1]-g.Q_true.iloc[-1])/g.Q_true.iloc[-1],signed_bias_pct=(100*(g.Q_pred-g.Q_true)/g.Q_true).mean(),optimistic_share=(g.Q_pred>g.Q_true).mean(),both_cross_error=abs(tc[0]-pc[0]) if eligible and len(tc) and len(pc) else np.nan))
    D=pd.DataFrame(rows);D.to_csv(ROOT/'tables/decision_seed_cells.csv',index=False);out=[]
    for (method,origin,seed),g in D.groupby(['method','origin','seed']):
        e=g[g.eligible];tp=int((e.true_event&e.pred_event).sum());fp=int((~e.true_event&e.pred_event).sum());fn=int((e.true_event&~e.pred_event).sum())
        out.append(dict(method=method,origin=origin,seed=seed,n_cells=len(g),n_eligible=len(e),true_events=tp+fn,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,last_APE=g.last_APE.mean(),signed_bias_pct=g.signed_bias_pct.mean(),optimistic_share=g.optimistic_share.mean(),both_cross_error=e.both_cross_error.mean(),censored_cross_error=abs(e.pred_cross_step-e.true_cross_step).mean()))
    S=pd.DataFrame(out);S.to_csv(ROOT/'tables/decision_seed_summary.csv',index=False);S.groupby(['method','origin']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(ROOT/'tables/decision_summary.csv',index=False)
    t=p[p.role.eq('test')].copy();t['APE']=100*abs(t.Q_pred-t.Q_true)/t.Q_true
    h=t.groupby(['method','origin','step','cell_id']).APE.mean().reset_index();rows=[]
    for (method,origin,step),g in h.groupby(['method','origin','step']):
        lo,hi=bootstrap_ci(g.APE);rows.append(dict(method=method,origin=origin,step=step,n_cells=len(g),Q_APE=g.APE.mean(),ci95_low=lo,ci95_high=hi))
    pd.DataFrame(rows).to_csv(ROOT/'tables/error_by_horizon.csv',index=False)

def main():
    metadata('final_analysis');p=canonical();p=combinations(p);p=selector(p)
    assert not p.duplicated(['method','fold','origin','role','seed','cell_id','row']).any()
    p.to_csv(ROOT/'results/canonical_predictions.csv',index=False);s=point_analysis(p);conformal(p);decisions(p)
    baseline=pd.read_csv(ROOT/'RESULTS_SUMMARY.csv');baseline=baseline[baseline.experiment.ne('round3_capacity')];pd.concat([baseline,s],ignore_index=True).to_csv(ROOT/'RESULTS_SUMMARY.csv',index=False)
    (ROOT/'audit/analysis_receipt.json').write_text(json.dumps(dict(passed=True,source_cells=104,primary_metric='Seed-error averaging within cell, then equalcells',interval_unit='Ah',interval_calibration='Additionaldisjointinnercalibrationcells, notmodel-selectionresiduals',levels=[.8,.9],crossing='LastmeasuredprefixQ>=80%Q0; firstfuturevalue<80%is crossing; no crossingright-censored atH+1. F1 computedpertrainingseed thenaveraged; eventcountsreported.',limitations='Intervals marginalperhorizon underexchangeability; no simultaneousorage-typeconditionalguarantee. Tailavailability changeshorizoncohort. Infiniteintervals areuninformative andfinitefractionmustbe reported.',primary_seedwise_metric=True),indent=2),encoding='utf8');print('ROUND3_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
