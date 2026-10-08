"""Learning, independent Oxford and prefix-robustness outcomes; no test selection."""
from statistics_r3 import *

def append_summary(frames):
    old=pd.read_csv(ROOT/'RESULTS_SUMMARY.csv');new=pd.concat(frames,ignore_index=True);experiments=set(new.experiment);old=old[~old.experiment.isin(experiments)];pd.concat([old,new],ignore_index=True).to_csv(ROOT/'RESULTS_SUMMARY.csv',index=False)

def learning():
    assert json.loads((ROOT/'audit/learning_receipt.json').read_text())['passed']
    p=pd.read_csv(ROOT/'results/learning_pred.csv');p=p[p.role.eq('test')].copy();p['APE']=100*abs(p.Q_pred-p.Q_true)/p.Q_true
    c=p.groupby(['learning_method','fraction','repeat','seed','cell_id']).APE.mean().groupby(['learning_method','fraction','cell_id']).mean().reset_index()
    best=pd.read_csv(ROOT/'tables/best_neural_by_fold_origin.csv');best=best[best.origin.eq(20)].set_index('fold').best_neural
    full=pd.read_csv(ROOT/'results/final_pred.csv');full=full[full.variant.eq('full')&full.origin.eq(20)&full.role.eq('test')].copy();full['APE']=100*abs(full.Q_pred-full.Q_true)/full.Q_true;copies=[]
    for fold,g in full.groupby('fold'):
        for name,family in [('xgb','xgb'),('best_neural',best[fold])]:
            q=g[g.family.eq(family)].groupby(['seed','cell_id']).APE.mean().groupby('cell_id').mean().rename('APE').reset_index();q['learning_method']=name;q['fraction']=1.;copies.append(q)
    c=pd.concat([c,*copies],ignore_index=True);assert c.groupby(['learning_method','fraction']).cell_id.nunique().eq(104).all();c.to_csv(ROOT/'tables/learning_cell_errors.csv',index=False)
    a=pd.read_csv(ROOT/'tables/completed_baseline_cells.csv');a=a[a.method.eq('ARIMA_AICc')&a.origin.eq(20)].set_index('cell_id').APE
    rows=[];pairs=[]
    for (method,fraction),g in c.groupby(['learning_method','fraction']):
        lo,hi=bootstrap_ci(g.APE);rows.append(dict(method=method,origin=20,experiment=f'learning_fraction_{fraction:g}',status='done',fraction=fraction,n_cells=len(g),Q_MAPE_equal_cell=g.APE.mean(),ci95_low=lo,ci95_high=hi))
        delta=g.set_index('cell_id').APE-a;dl,dh=bootstrap_ci(delta);pairs.append(dict(family='learning_all_fractions_methods_vs_ARIMA_all6',method=method,fraction=fraction,n_cells=len(delta),difference_pp=delta.mean(),ci95_low=dl,ci95_high=dh,p_raw=wilcoxon(delta).pvalue if np.any(delta) else 1.))
    s=pd.DataFrame(rows);s.to_csv(ROOT/'tables/learning_summary.csv',index=False);t=pd.DataFrame(pairs);t['p_Holm']=holm(t.p_raw);t['family_size']=len(t);t.to_csv(ROOT/'tables/learning_pairs.csv',index=False)
    runs=pd.read_csv(ROOT/'results/learning_runs.csv');runs.groupby(['learning_method','fraction','repeat']).agg(mean_fitting_cells=('train_cells','mean'),mean_effective_cells=('cells_contributing_examples','mean'),mean_teacher_examples=('teacher_examples','mean'),mean_rollout_segments=('rollout_segments','mean')).reset_index().to_csv(ROOT/'tables/learning_effective_data.csv',index=False)
    return s

def oxford():
    assert json.loads((ROOT/'audit/oxford_models_receipt.json').read_text())['passed']
    p=pd.read_csv(ROOT/'results/oxford_pred.csv');p=p[p.role.eq('test')].rename(columns={'family':'method'});p=p[['method','seed','cell_id','row','Q_true','Q_pred']]
    controls=pd.concat([pd.read_csv(ROOT/'results/oxford_controls_pred.csv'),pd.read_csv(ROOT/'results/oxford_chronos_pred.csv')],ignore_index=True);controls['seed']=0
    p=pd.concat([p,controls[p.columns]],ignore_index=True);assert not p.duplicated(['method','seed','cell_id','row']).any();p['origin']=20;p['APE']=100*abs(p.Q_pred-p.Q_true)/p.Q_true
    c=p.groupby(['method','seed','cell_id']).APE.mean().groupby(['method','cell_id']).mean().reset_index();c['origin']=20;assert c.groupby('method').cell_id.nunique().eq(8).all();c.to_csv(ROOT/'tables/oxford_cell_errors.csv',index=False)
    s=summarize(c,'independent_Oxford');s.to_csv(ROOT/'tables/oxford_summary.csv',index=False);t=paired(c,[(20,m,'ARIMA_AICc') for m in sorted(set(c.method)-{'ARIMA_AICc'})],'Oxford_all_methods_vs_ARIMA');t.to_csv(ROOT/'tables/oxford_pairs.csv',index=False)
    return s

def robustness():
    assert json.loads((ROOT/'audit/robustness_arima_receipt.json').read_text())['passed'] and json.loads((ROOT/'audit/robustness_models_receipt.json').read_text())['passed']
    p=pd.concat([pd.read_csv(ROOT/'results/robustness_arima_pred.csv'),pd.read_csv(ROOT/'results/robustness_model_pred.csv')],ignore_index=True)
    assert not p.duplicated(['method','model_seed','job_id','cell_id','row']).any();p['APE']=100*abs(p.Q_pred-p.Q_true)/p.Q_true
    c=p.groupby(['method','noise_sigma_Q0','drop_fraction','model_seed','noise_seed','cell_id']).APE.mean().groupby(['method','noise_sigma_Q0','drop_fraction','noise_seed','cell_id']).mean().groupby(['method','noise_sigma_Q0','drop_fraction','cell_id']).mean().reset_index()
    assert c.groupby(['method','noise_sigma_Q0','drop_fraction']).cell_id.nunique().eq(104).all();c.to_csv(ROOT/'tables/robustness_cell_errors.csv',index=False)
    rows=[];tests=[]
    for (method,sigma,drop),g in c.groupby(['method','noise_sigma_Q0','drop_fraction']):
        lo,hi=bootstrap_ci(g.APE);rows.append(dict(method=method,origin=20,experiment=f'robustness_sigma{sigma:g}_drop{drop:g}',status='done',noise_sigma_Q0=sigma,drop_fraction=drop,n_cells=len(g),Q_MAPE_equal_cell=g.APE.mean(),ci95_low=lo,ci95_high=hi))
        if sigma==drop==0:continue
        base=c[c.method.eq(method)&c.noise_sigma_Q0.eq(0)&c.drop_fraction.eq(0)].set_index('cell_id').APE;delta=g.set_index('cell_id').APE-base;dl,dh=bootstrap_ci(delta)
        tests.append(dict(family=f'robustness_vs_clean_{method}_all8',method=method,noise_sigma_Q0=sigma,drop_fraction=drop,n_cells=len(delta),difference_pp=delta.mean(),ci95_low=dl,ci95_high=dh,p_raw=wilcoxon(delta).pvalue if np.any(delta) else 1.))
    s=pd.DataFrame(rows);s.to_csv(ROOT/'tables/robustness_summary.csv',index=False);t=pd.DataFrame(tests)
    for family,g in t.groupby('family'):t.loc[g.index,'p_Holm']=holm(g.p_raw);t.loc[g.index,'family_size']=len(g)
    t.to_csv(ROOT/'tables/robustness_pairs.csv',index=False)
    # Every clean scenario must reproduce the original fixed predictor at the same cell/seed/row.
    main=pd.read_csv(ROOT/'results/final_pred.csv');main=main[main.origin.eq(20)&main.variant.eq('full')&main.role.eq('test')].rename(columns={'family':'method','seed':'model_seed'})
    clean=p[p.noise_sigma_Q0.eq(0)&p.drop_fraction.eq(0)&p.method.isin(FAMILIES)]
    m=clean.merge(main[['method','model_seed','cell_id','row','Q_pred']],on=['method','model_seed','cell_id','row'],validate='one_to_one',suffixes=('_robust','_main'))
    assert len(m)==len(clean) and np.allclose(m.Q_pred_robust,m.Q_pred_main,rtol=1e-6,atol=2e-6)
    runs=pd.read_csv(ROOT/'results/robustness_arima_runs.csv');runs.groupby(['noise_sigma_Q0','drop_fraction','status']).size().rename('n').reset_index().to_csv(ROOT/'tables/robustness_arima_statuses.csv',index=False)
    return s

def main():
    metadata('extra_analysis');summaries=[learning(),oxford(),robustness()];append_summary(summaries)
    (ROOT/'audit/extra_analysis_receipt.json').write_text(json.dumps(dict(passed=True,learning_cells=104,Oxford_cells=8,robustness_cells=104,bootstrap_draws=10000,bootstrap_seed=20261005,clean_model_replay_verified=True,limits='Learningconditionalonfold-tunedstructures; Oxfordsmall8cellindependentprotocolnotbroadchemistryguarantee; noisesyntheticprefixsensitivitynotempiricalfieldvalidation.'),indent=2),encoding='utf8');print('EXTRA_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
