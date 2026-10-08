"""Idle-suite1-thread deployment measurements and inner-validation frontiers."""
from r3 import *
import pickle,subprocess,gc

def fits_active():
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '(run_models|learning_curve|oxford_models|robustness)\\.py' } | Select-Object -ExpandProperty ProcessId"
    r=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,text=True);return bool(r.stdout.strip())
def job_rows():
    files=['search','final','resistance','static_alias'];frames=[]
    for stage in files:
        r=pd.read_csv(ROOT/'results'/f'{stage}_runs.csv');r['stage_file']=stage;frames.append(r)
    return pd.concat(frames,ignore_index=True)
def measure(row):
    import torch,exp_lib as X
    torch.set_num_threads(1);payload=pickle.loads((ROOT/'checkpoints'/f'{row.job_id}.pkl').read_bytes());features=payload['features'];L=payload['cfg']['L'];origin=int(row.origin)
    frame=split(int(row.fold))['validation'];frame,_=information(frame,str(row.variant));frame=hold_resistance(frame,origin) if 'Re_history' in features else frame
    # Representative validation cells by regime, fixed without test-performance choices.
    examples=[]
    for _,g in frame.groupby('age_type'):
        cid=sorted(g.cell_id.unique())[0];q=ordered(g[g.cell_id.eq(cid)]);examples.append(q[features+['Q']].to_numpy(float)[origin-L:origin])
    if payload['kind']=='neural':
        model=X.make_model(payload['family'],len(features)+1,payload['cfg']).eval();model.load_state_dict(payload['state']);times=[]
        for e in examples:
            x=payload['sx'].transform(e).astype(np.float32);times.append(np.median([X.cpu_latency_ms(model,L,len(features)+1,example=x) for _ in range(3)]))
        params=X.count_params(model);nodes=np.nan
    else:
        model=payload['model'];times=[]
        for e in examples:
            x=e.ravel()[None];times.append(np.median([benchmark(lambda:model.predict(x)) for _ in range(3)]))
        params=np.nan;nodes=sum(t.tree_.node_count for t in model.estimators_) if row.family=='rf' else len(model.get_booster().trees_to_dataframe())
    return dict(job_id=row.job_id,stage=row.stage_file,family=row.family,variant=row.variant,fold=row.fold,origin=origin,seed=row.seed,quiet_latency_cpu1_ms=float(np.median(times)),regime_examples=len(examples),parameters=params,tree_nodes=nodes,serialized_bytes=row.serialized_bytes,weights_bytes=row.weights_bytes,fit_wall_seconds=row.fit_wall_seconds,fit_cpu_seconds=row.fit_cpu_seconds,val_Q_MAPE_equal_cell=row.val_Q_MAPE_equal_cell)
def nondominated(g,axes):
    x=g[axes].to_numpy(float);return np.array([not np.any(np.all(x<=v,axis=1)&np.any(x<v,axis=1)) for v in x])
def main():
    assert not fits_active(),'Do not call loaded timings quiet';metadata('quiet_resources');time.sleep(10)
    rp=ROOT/'tables/quiet_resources.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set();allrows=job_rows()
    for row in allrows.itertuples():
        if row.job_id in done:continue
        if row.stage_file=='static_alias':
            old=pd.read_csv(rp);match=old[old.job_id.eq(row.alias_of)].iloc[0].to_dict();match.update(job_id=row.job_id,stage='static_alias',variant='static',fit_wall_seconds=0.,fit_cpu_seconds=0.)
            result=match
        else:result=measure(row)
        append(rp,pd.DataFrame([result]));done.add(row.job_id);print('QUIET_RESOURCE',len(done),'/',len(allrows),row.job_id,flush=True);gc.collect()
    assert len(done)==len(allrows)
    r=pd.read_csv(rp);search=r[r.stage.eq('search')].copy();axes=['val_Q_MAPE_equal_cell','quiet_latency_cpu1_ms','serialized_bytes'];search['frontier_accuracy_size_latency']=False;search['frontier_plus_training_cpu']=False
    primary=r[r.stage.eq('final')&r.variant.eq('full')&r.origin.eq(20)]
    assert len(primary)==75
    primary.groupby('family').agg(
        fitted_models=('job_id','count'),folds=('fold','nunique'),
        mean_fit_cpu_seconds=('fit_cpu_seconds','mean'),total_fit_cpu_seconds=('fit_cpu_seconds','sum'),
        mean_fit_wall_seconds=('fit_wall_seconds','mean'),total_fit_wall_seconds=('fit_wall_seconds','sum'),
        median_quiet_latency_cpu1_ms=('quiet_latency_cpu1_ms','median'),
        minimum_quiet_latency_cpu1_ms=('quiet_latency_cpu1_ms','min'),maximum_quiet_latency_cpu1_ms=('quiet_latency_cpu1_ms','max'),
        mean_serialized_bytes=('serialized_bytes','mean'),minimum_serialized_bytes=('serialized_bytes','min'),maximum_serialized_bytes=('serialized_bytes','max'),
        mean_parameters=('parameters','mean'),mean_tree_nodes=('tree_nodes','mean')
    ).reset_index().to_csv(ROOT/'tables/quiet_primary_resources.csv',index=False)
    search.groupby(['family','fold']).agg(
        search_fits=('job_id','count'),search_fit_cpu_seconds=('fit_cpu_seconds','sum'),
        search_fit_wall_seconds=('fit_wall_seconds','sum')
    ).reset_index().to_csv(ROOT/'tables/search_training_costs.csv',index=False)
    for fold,g in search.groupby('fold'):
        search.loc[g.index,'frontier_accuracy_size_latency']=nondominated(g,axes);search.loc[g.index,'frontier_plus_training_cpu']=nondominated(g,axes+['fit_cpu_seconds'])
    search.to_csv(ROOT/'tables/validation_trial_frontiers.csv',index=False)
    # Descriptive cross-fold mean at identical random configuration; not a test-selected config.
    trials=pd.read_csv(ROOT/'results/search_runs.csv')[['job_id','trial']];avg=search.merge(trials,on='job_id').groupby(['family','trial'])[axes+['fit_cpu_seconds']].mean().reset_index();avg['frontier']=nondominated(avg,axes);avg.to_csv(ROOT/'tables/validation_mean_frontier.csv',index=False)
    (ROOT/'audit/quiet_resources_receipt.json').write_text(json.dumps(dict(passed=True,models_measured=len(done),threads=1,warmup_calls=20,blocks=5,calls_per_block=20,whole_pass_repeats=3,real_validation_regime_examples=True,scope='No ownfit/robustness jobs active at passstart. OSambientload notfullyisolated; CPUlaptop, notBMSmicrocontroller.',frontier='Inner-validation error/quietlatency/serializedsize, withinfold. Four-objective sensitivity adds fittingCPUtime. Cross-foldmeanfrontierdescriptive only; no outertestselection.',serialization='Owntrained deployable Python payload including scalers where applicable; tree-nodecount notneuralparametercount.'),indent=2),encoding='utf8');print('QUIET_RESOURCES_COMPLETE',flush=True)
if __name__=='__main__':main()
