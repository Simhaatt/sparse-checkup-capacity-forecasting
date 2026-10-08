"""Requirement-wide numerical, membership, checkpoint and scaler audit."""
from r3 import *
import run_models as M
import pickle,gc
from sklearn.preprocessing import StandardScaler

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--core-only',action='store_true');args=parser.parse_args()
    metadata('core_verification' if args.core_only else 'completion_verification');checks=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
        if not passed:raise AssertionError(name+': '+str(detail))
    plan=prepare();d=data();long=cohort();parts=pd.read_csv(ROOT/'audit/cell_partitions.csv');foldmap=pd.read_csv(ROOT/'inputs/grouped_cv_folds.csv').set_index('cell_id').fold
    check('104sourcecells',len(long)==104);check('capacityrawcheck',json.loads((ROOT/'audit/prior_raw_capacity_receipt.json').read_text())['passed'])
    for fold,g in parts.groupby('fold'):
        check(f'partition_unique_f{fold}',not g.cell_id.duplicated().any() and set(g.cell_id)==set(d.cell_id))
        check(f'validation15pct_f{fold}',int(g.role.eq('validation').sum())==round(.15*int(g.role.ne('test').sum())))
        check(f'calibration20_f{fold}',g.role.eq('calibration').sum()==20)
        check(f'outerfold_f{fold}',set(g[g.role.eq('test')].cell_id)==set(foldmap[foldmap.eq(fold)].index))
    truth=[]
    for cid,g0 in d.groupby('cell_id'):
        g=ordered(g0);truth.extend(dict(cell_id=cid,row=i,Q_true_source=float(q)) for i,q in enumerate(g.Q))
    truth=pd.DataFrame(truth);sizes=d.groupby('cell_id').size();w=pd.read_csv(ROOT/'tables/fold_selected_settings.csv')
    expected={'search':list(M.search_jobs(plan)),'final':list(M.final_jobs(w)),'resistance':list(M.resistance_jobs(w))}
    if args.core_only:expected.pop('resistance')
    scaler_cache={};allruns=[]
    import torch,exp_lib as X
    torch.set_num_threads(1)
    for stage,jobs in expected.items():
        runs=pd.read_csv(ROOT/'results'/f'{stage}_runs.csv');p=pd.read_csv(ROOT/'results'/f'{stage}_pred.csv');ids={j['job_id'] for j in jobs}
        check(stage+'_job_grid',len(runs)==len(jobs) and not runs.job_id.duplicated().any() and set(runs.job_id)==ids and set(p.job_id)==ids)
        check(stage+'_capacity_only',not any(c.startswith('Re_pred') for c in p.columns));check(stage+'_positive_finite',np.isfinite(p[['Q_true','Q_pred']]).all().all() and p[['Q_true','Q_pred']].gt(0).all().all())
        matched=p.merge(truth,on=['cell_id','row'],validate='many_to_one');check(stage+'_truth',len(matched)==len(p) and np.allclose(matched.Q_true,matched.Q_true_source,atol=2e-6,rtol=1e-7))
        check(stage+'_unique_prediction_keys',not p.duplicated(['job_id','role','cell_id','row']).any())
        for r in runs.itertuples():
            g=p[p.job_id.eq(r.job_id)];membership=parts[parts.fold.eq(r.fold)].set_index('cell_id').role
            actual_validation=equal_cell(g[g.role.eq('validation')])
            check(r.job_id+'_saved_validation_score',np.isclose(actual_validation,r.val_Q_MAPE_equal_cell,rtol=0,atol=1e-6),'Absolute percentage-point tolerance for float32-to-CSV round trip; see validation_score_roundtrip_receipt.json')
            check(r.job_id+'_role',g.role.eq(g.cell_id.map(membership)).all())
            for role in (['validation'] if stage=='search' else ['validation','calibration','test']):
                cells=set(membership[membership.eq(role)].index)&long;actual=g[g.role.eq(role)];n=int((sizes[sizes.index.isin(cells)]-r.origin).sum())
                check(r.job_id+'_'+role+'_coverage',len(actual)==n and set(actual.cell_id)==cells and actual.row.ge(r.origin).all())
            path=ROOT/'checkpoints'/f'{r.job_id}.pkl';check(r.job_id+'_checkpoint_hash',sha(path)==r.checkpoint_sha256);payload=pickle.loads(path.read_bytes())
            check(r.job_id+'_Qonly_shape',payload.get('origin')==r.origin and payload['features'][-1:]!=['Q'])
            if payload['kind']=='neural':
                key=(r.fold,payload['cfg']['L'],r.variant)
                if key not in scaler_cache:
                    tr=information(split(r.fold)['fit'],r.variant)[0];xx,yy=X.build_sequences(tr,payload['features'],payload['cfg']['L']);nf=xx.shape[-1]
                    sx=StandardScaler().fit(xx.reshape(-1,nf));sy=StandardScaler().fit(yy);scaler_cache[key]=(sx.mean_,sx.scale_,sy.mean_,sy.scale_)
                expected_scalers=scaler_cache[key];actual=(payload['sx'].mean_,payload['sx'].scale_,payload['sy'].mean_,payload['sy'].scale_)
                check(r.job_id+'_scaler_training_only',all(np.allclose(a,b,atol=1e-9,rtol=1e-8) for a,b in zip(actual,expected_scalers)))
                model=X.make_model(payload['family'],len(payload['features'])+1,payload['cfg']);check(r.job_id+'_parameters',X.count_params(model)==r.parameters);check(r.job_id+'_output_Q',len(payload['sy'].mean_)==1 and payload['state'][f'head.{len(model.head)-1}.weight'].shape[0]==1);del model
            del payload;gc.collect()
        allruns.append(runs)
    search=pd.read_csv(ROOT/'results/search_runs.csv');check('20trials_every_family_fold',search.groupby(['fold','family']).trial.nunique().eq(20).all())
    for row in w.itertuples():
        g=search[search.fold.eq(row.fold)&search.family.eq(row.family)].sort_values(['val_Q_MAPE_equal_cell','trial']);check(f'winner_f{row.fold}_{row.family}',row.job_id==g.iloc[0].job_id)
    saved_search=pd.read_csv(ROOT/'results/search_pred.csv');saved_search['APE']=100*abs(saved_search.Q_pred-saved_search.Q_true)/saved_search.Q_true
    csv_scores=saved_search.groupby(['job_id','cell_id']).APE.mean().groupby('job_id').mean()
    for row in w.itertuples():
        g=search[search.fold.eq(row.fold)&search.family.eq(row.family)].copy();g['csv_validation_score']=g.job_id.map(csv_scores)
        actual_winner=g.sort_values(['csv_validation_score','trial']).iloc[0].job_id
        check(f'CSV_rederived_winner_f{row.fold}_{row.family}',row.job_id==actual_winner)
    del saved_search
    compact=[t['cfg'] for t in plan['trials'] if t['family']=='compact_lstm'];check('compact_spec',all(c['hidden']<=64 and c['layers']==1 for c in compact))
    aliases=pd.read_csv(ROOT/'results/static_alias_runs.csv');check('75static_aliases',len(aliases)==75 and not aliases.job_id.duplicated().any())
    for row in aliases.itertuples():check(row.job_id+'_exact_alias',sha(ROOT/'checkpoints'/f'{row.job_id}.pkl')==sha(ROOT/'checkpoints'/f'{row.alias_of}.pkl'))
    alias_predictions=pd.read_csv(ROOT/'results/static_alias_pred.csv')
    final_predictions=pd.read_csv(ROOT/'results/final_pred.csv');mapping=aliases.set_index('alias_of').job_id.to_dict()
    source_predictions=final_predictions[final_predictions.job_id.isin(mapping)].copy();source_predictions['job_id']=source_predictions.job_id.map(mapping)
    keys=['job_id','role','cell_id','row']
    check('static_alias_prediction_keys',not alias_predictions.duplicated(keys).any() and set(alias_predictions.job_id)==set(aliases.job_id))
    alias_comparison=alias_predictions.merge(source_predictions[keys+['Q_true','Q_pred']],on=keys,validate='one_to_one',suffixes=('_alias','_source'))
    check('static_alias_prediction_identity',len(alias_comparison)==len(alias_predictions)==len(source_predictions) and np.allclose(alias_comparison.Q_true_alias,alias_comparison.Q_true_source,rtol=0,atol=1e-12) and np.allclose(alias_comparison.Q_pred_alias,alias_comparison.Q_pred_source,rtol=0,atol=1e-12))
    del alias_predictions,final_predictions,source_predictions,alias_comparison
    if args.core_only:
        report=dict(passed=all(c['passed'] for c in checks),checks=checks,scope='Completed search500/core375/static75 grid, source truth, role membership, validation scores, checkpoint hashes, neural training-only scalers, compact limits and validation winners. This is not full round completion.')
        (ROOT/'audit/core_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');pd.DataFrame(checks).to_csv(ROOT/'audit/core_checks.csv',index=False)
        print('CORE_NUMERICAL_VERIFICATION_PASS',len(checks),flush=True);return
    aligned=pd.read_csv(ROOT/'results/aligned_re_v2.csv');valid=aligned[aligned.Re_mask.eq(1)];check('raw228files',json.loads((ROOT/'audit/aligned_re_receipt_v2.json').read_text())['raw_cells_obtained']==228);check('backward14day',valid.lag_days.between(0,14).all());check('missing_not_imputed',aligned[aligned.Re_mask.eq(0)].Re_aligned.isna().all())
    check('aligned_no_future_labels',pd.read_csv(ROOT/'tables/aligned_macro_unavailable.csv').aligned_future_Re_labels.eq(0).all())
    for name in ['analysis_receipt.json','extra_analysis_receipt.json','learning_receipt.json','oxford_models_receipt.json','robustness_arima_receipt.json','robustness_models_receipt.json','quiet_resources_receipt.json','plots_receipt.json','semiempirical_receipt_v3.json','macro_diagnostic_receipt.json','chronos_tiny_receipt.json']:
        check(name,json.loads((ROOT/'audit'/name).read_text()).get('passed',False))
    # Verify the full extra-study grids, rather than accepting their receipt names alone.
    lr=pd.read_csv(ROOT/'results/learning_runs.csv');lp=pd.read_csv(ROOT/'results/learning_pred.csv');subsets=pd.read_csv(ROOT/'audit/learning_subsets.csv')
    check('learning180grid',len(lr)==180 and not lr.job_id.duplicated().any() and set(lr.job_id)==set(lp.job_id))
    check('learning_structure',lr.groupby(['fold','learning_method','fraction']).size().eq(9).all() and set(lr.fraction)=={.25,.5} and set(lr.seed)==set(SEEDS) and set(lr['repeat'])=={0,1,2})
    for row in lr.itertuples():
        check(row.job_id+'_hash',sha(ROOT/'checkpoints'/f'{row.job_id}.pkl')==row.checkpoint_sha256)
        cells=set(subsets[subsets.fold.eq(row.fold)&subsets['repeat'].eq(row.repeat)&subsets.fraction.eq(row.fraction)].cell_id);fit=split(row.fold)['fit'];check(row.job_id+'_subset_only',cells<=set(fit.cell_id) and len(cells)==row.train_cells)
        payload=pickle.loads((ROOT/'checkpoints'/f'{row.job_id}.pkl').read_bytes())
        if payload['kind']=='neural':
            tr=fit[fit.cell_id.isin(cells)];xx,yy=X.build_sequences(tr,F7,payload['cfg']['L']);sx=StandardScaler().fit(xx.reshape(-1,8));sy=StandardScaler().fit(yy)
            check(row.job_id+'_subset_scalers',np.allclose(sx.mean_,payload['sx'].mean_) and np.allclose(sy.mean_,payload['sy'].mean_))
        del payload
    check('learning_all104',pd.read_csv(ROOT/'tables/learning_cell_errors.csv').groupby(['learning_method','fraction']).cell_id.nunique().eq(104).all())
    import oxford_models as O
    ox=O.oxford();orr=pd.read_csv(ROOT/'results/oxford_runs.csv');op=pd.read_csv(ROOT/'results/oxford_pred.csv');check('Oxford120grid',len(orr)==120 and not orr.job_id.duplicated().any() and set(orr.job_id)==set(op.job_id))
    for row in orr.itertuples():
        path=ROOT/'checkpoints'/f'{row.job_id}.pkl';check(row.job_id+'_hash',sha(path)==row.checkpoint_sha256);g=op[op.job_id.eq(row.job_id)]
        check(row.job_id+'_outercell',set(g[g.role.eq('test')].cell_id)=={f'Cell{row.fold}'});check(row.job_id+'_validationcell',set(g[g.role.eq('selection')].cell_id)=={f'Cell{row.fold%8+1}'})
        true=ox.sort_values(['cell_id','timestamp_s']).copy();true['row']=true.groupby('cell_id').cumcount();match=g.merge(true[['cell_id','row','Q']],on=['cell_id','row'],validate='many_to_one');check(row.job_id+'_truth',len(match)==len(g) and np.allclose(match.Q_true,match.Q,atol=2e-6))
        payload=pickle.loads(path.read_bytes())
        if payload['kind']=='neural':
            tr,_,_=O.oxsplit(row.fold);xx,yy=X.build_sequences(tr,[],payload['cfg']['L']);sx=StandardScaler().fit(xx.reshape(-1,1));sy=StandardScaler().fit(yy);check(row.job_id+'_sixcell_scalers',np.allclose(sx.mean_,payload['sx'].mean_) and np.allclose(sy.mean_,payload['sy'].mean_))
        del payload
    check('Oxford_all8methods',pd.read_csv(ROOT/'tables/oxford_cell_errors.csv').groupby('method').cell_id.nunique().eq(8).all())
    rr=pd.read_csv(ROOT/'results/robustness_arima_runs.csv');jobs=pd.read_csv(ROOT/'audit/robustness_jobs.csv');rp=pd.read_csv(ROOT/'results/robustness_arima_pred.csv');rm=pd.read_csv(ROOT/'results/robustness_model_pred.csv')
    check('robust2600grid',len(rr)==2600 and set(rr.job_id)==set(jobs.job_id) and not rr.job_id.duplicated().any());check('robust75models',len(pd.read_csv(ROOT/'results/robustness_model_runs.csv'))==75)
    for pred,name in [(rp,'robust_ARIMA'),(rm,'robust_learned')]:
        match=pred.merge(truth,on=['cell_id','row'],validate='many_to_one');check(name+'_unchanged_truth',len(match)==len(pred) and np.allclose(match.Q_true,match.Q_true_source,atol=2e-6))
        check(name+'_scenario_cells',pred.groupby(['method','noise_sigma_Q0','drop_fraction']).cell_id.nunique().eq(104).all())
        check(name+'_unique',not pred.duplicated(['method','model_seed','job_id','cell_id','row']).any())
    check('robust_all104errors',pd.read_csv(ROOT/'tables/robustness_cell_errors.csv').groupby(['method','noise_sigma_Q0','drop_fraction']).cell_id.nunique().eq(104).all())
    # Independent calibration-role radii: recompute every rank from canonical predictions.
    canonical=pd.read_csv(ROOT/'results/canonical_predictions.csv');radii=pd.read_csv(ROOT/'tables/conformal_radii.csv')
    calibration=canonical[canonical.role.eq('calibration')]
    cache={key:np.sort(abs(g.Q_true-g.Q_pred).to_numpy()) for key,g in calibration.groupby(['method','fold','origin','seed','step'])}
    for row in radii.itertuples():
        key=(row.method,row.fold,row.origin,row.seed,row.step)
        values=cache.get(key,np.array([],float));rank=int(np.ceil((len(values)+1)*row.nominal));expected_radius=values[rank-1] if rank<=len(values) and len(values) else np.inf
        check('conformal:'+':'.join(map(str,key))+':'+str(row.nominal),len(values)==row.n_calibration and rank==row.quantile_rank and (np.isinf(expected_radius) and np.isinf(row.radius_Ah) or np.isclose(expected_radius,row.radius_Ah)))
    summary=pd.read_csv(ROOT/'RESULTS_SUMMARY.csv');check('summary_unique',not summary.duplicated(['method','origin','experiment']).any());check('source_summary_cohort',summary[~summary.experiment.eq('independent_Oxford')].n_cells.eq(104).all())
    manifest=pd.read_csv(ROOT/'tables/figure_manifest.csv');check('figure_files',all((ROOT/p).exists() for p in [*manifest.png,*manifest.pdf]))
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,scope='Allmainmodeljobgrids, allneuraltraining-onlyscalers/checkpointhashes, selectedinnerwinners, predictiontruth/membership/coverage, everycalibrationradius, extraexperimentreceipts andfigureexistence. VisualQA andfinalMDcontent review separatelyrequired.')
    (ROOT/'audit/completion_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');pd.DataFrame(checks).to_csv(ROOT/'audit/completion_checks.csv',index=False);print('ROUND3_NUMERICAL_VERIFICATION_PASS',len(checks),flush=True)
if __name__=='__main__':main()
