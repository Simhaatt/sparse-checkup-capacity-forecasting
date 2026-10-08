"""Resumable equal-budget, cell-disjoint CPU search; never score outer test trials."""
from study import *
from concurrent.futures import ProcessPoolExecutor
import pickle,copy,traceback

def fit_job(job):
    import torch
    import exp_lib as X
    torch.set_num_threads(1);X.COMMON_START=job['origin']
    parts=split(job['fold']);parts={k:information(v,job['information'])[0] for k,v in parts.items()}
    features=F7 if job['information']=='scheduled' else FIXED if job['information']=='prefix_fixed' else []
    family=job['family'];cfg=copy.deepcopy(job['cfg']);cfg['L']=min(cfg['L'],job['origin'])
    tr,va=parts['fit'],parts['selection'];t=time.perf_counter();cpu=time.process_time()
    path=ROOT/'checkpoints'/f"{job['job_id']}.pkl"
    if family in ['compact_lstm','gru','large_lstm']:
        fam='gru' if family=='gru' else 'lstm'
        row,_,(m,sx,sy)=X.train_eval(tr,va,None,fam,cfg,features,job['seed'],job['schedule'])
        fit_wall=time.perf_counter()-t;fit_cpu=time.process_time()-cpu
        val=predict_neural(X,m,sx,sy,va,features,cfg['L'])
        g=ordered(va[va.cell_id.eq(sorted(va.cell_id.unique())[0])]);w=sx.transform(g[features+['Q']].to_numpy(float)[job['origin']-cfg['L']:job['origin']]).astype(np.float32)
        latency=X.cpu_latency_ms(m,cfg['L'],len(features)+1,example=w)
        payload=dict(kind='neural',family=fam,cfg=cfg,features=features,information=job['information'],state={k:v.cpu() for k,v in m.state_dict().items()},sx=sx,sy=sy,origin=job['origin'])
        cost=dict(parameters=X.count_params(m),weights_bytes=X.serialized_bytes(m),latency_cpu1_ms=latency)
    else:
        xx,yy=X.build_sequences(tr,features,cfg['L']);xx=xx.reshape(len(xx),-1);m=tree_model(family,cfg,job['seed']);m.fit(xx,yy.ravel())
        fit_wall=time.perf_counter()-t;fit_cpu=time.process_time()-cpu
        val=predict_tree(m,va,features,cfg['L'],job['origin'])
        g=ordered(va[va.cell_id.eq(sorted(va.cell_id.unique())[0])]);w=g[features+['Q']].to_numpy(float)[job['origin']-cfg['L']:job['origin']].ravel()[None]
        latency=benchmark(lambda:m.predict(w))
        payload=dict(kind='tree',family=family,cfg=cfg,features=features,information=job['information'],model=m,origin=job['origin'])
        cost=dict(tree_nodes=sum(e.tree_.node_count for e in m.estimators_) if family=='rf' else len(m.get_booster().trees_to_dataframe()),weights_bytes=len(pickle.dumps(m)),latency_cpu1_ms=latency)
    with path.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    row={k:v for k,v in job.items() if k!='cfg'};row.update(cfg_json=json.dumps(cfg,sort_keys=True),val_Q_MAPE_equal_cell=equal_cell(val),fit_wall_seconds=fit_wall,fit_cpu_seconds=fit_cpu,serialized_bytes=path.stat().st_size,checkpoint_sha256=sha(path),train_cells=tr.cell_id.nunique(),selection_cells=va.cell_id.nunique(),**cost)
    preds=[]
    roles=['selection'] if job['stage']=='search' else ['selection','calibration','test']
    for role in roles:
        frame=parts[role];frame=frame[frame.cell_id.isin(cohort(data()))]
        p=val.copy() if role=='selection' else predict_neural(X,m,sx,sy,frame,features,cfg['L']) if payload['kind']=='neural' else predict_tree(m,frame,features,cfg['L'],job['origin'])
        for k in ['job_id','family','fold','seed','origin','information','stage']:p[k]=job[k]
        p['role']=role;preds.append(p)
    return row,pd.concat(preds,ignore_index=True)

def search_jobs(plan):
    for f in range(1,6):
        # Interleave families to give early evidence of failures/fit costs across all five.
        for trial in range(1,21):
            for family in FAMILIES:
                cfg=next(t['cfg'] for t in plan['trials'] if t['family']==family and t['trial']==trial)
                yield dict(job_id=f'search_f{f}_{family}_t{trial:02d}',stage='search',family=family,fold=f,trial=trial,cfg=cfg,seed=42,origin=20,information='scheduled',schedule='search')

def select_winners():
    r=pd.read_csv(ROOT/'results/search_runs.csv');out=[]
    for (fold,family),g in r.groupby(['fold','family']):
        assert len(g)==20 and set(g.trial)==set(range(1,21))
        w=g.sort_values(['val_Q_MAPE_equal_cell','trial'],kind='mergesort').iloc[0];out.append(w)
    assert len(out)==25
    winners=pd.DataFrame(out);winners.to_csv(ROOT/'tables/fold_selected_settings.csv',index=False)
    receipt=dict(passed=True,selection='Equal-cell Q-MAPE of inner selection only',trials_per_family_per_fold=20,selected_configs=25,search_rows=500,source_sha256=sha(ROOT/'results/search_runs.csv'),partitions_sha256=sha(ROOT/'audit/cell_partitions.csv'))
    (ROOT/'audit/selection_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return winners

def final_jobs(winners):
    for w in winners.itertuples():
        cfg=json.loads(w.cfg_json)
        for origin in [20,10,15]:
            modes=['scheduled','prefix_fixed','history_only'] if origin==20 else ['scheduled']
            for mode in modes:
                for seed in SEEDS:
                    yield dict(job_id=f'final_o{origin}_f{w.fold}_{w.family}_{mode}_s{seed}',stage='final',family=w.family,fold=w.fold,trial=int(w.trial),cfg=cfg,seed=seed,origin=origin,information=mode,schedule='final')

def run_stage(jobs,name):
    path=ROOT/'results'/f'{name}_runs.csv';pp=ROOT/'results'/f'{name}_pred.csv'
    done=set(pd.read_csv(path).job_id) if path.exists() else set();jobs=list(jobs)
    remaining=[j for j in jobs if j['job_id'] not in done]
    # A saved completion row requires its prediction rows and intact checkpoint.
    if done:
        p=pd.read_csv(pp);assert done<=set(p.job_id)
        for r in pd.read_csv(path).itertuples():assert sha(ROOT/'checkpoints'/f'{r.job_id}.pkl')==r.checkpoint_sha256
    print('STAGE',name,'complete',len(done),'remaining',len(remaining),'expected',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for row,pred in pool.map(fit_job,remaining):
            # Parent alone writes aggregate files. Predictions precede completion markers.
            if pp.exists():
                old=pd.read_csv(pp)
                if old.job_id.eq(row['job_id']).any():old[~old.job_id.eq(row['job_id'])].to_csv(pp,index=False)
            append(pp,pred);append(path,pd.DataFrame([row]));done.add(row['job_id'])
            print('DONE',name,len(done),'/',len(jobs),row['job_id'],'val',round(row['val_Q_MAPE_equal_cell'],5),'fit_s',round(row['fit_wall_seconds'],2),flush=True)
    assert len(done)==len(jobs)

def smoke():
    plan=prepare();j=next(search_jobs(plan));j.update(job_id='smoke_fixed',schedule='smoke',information='prefix_fixed')
    row,p=fit_job(j)
    import torch,exp_lib as X
    torch.set_num_threads(1)
    payload=pickle.loads((ROOT/'checkpoints/smoke_fixed.pkl').read_bytes());m=X.make_model(payload['family'],len(payload['features'])+1,payload['cfg']);m.load_state_dict(payload['state'])
    test=split(1)['test'];test=test[test.cell_id.isin(cohort(data()))]
    x,features=information(test,'prefix_fixed');a=predict_neural(X,m,payload['sx'],payload['sy'],x,features,payload['cfg']['L'])
    y=test.copy();mask=y.groupby('cell_id').cumcount().ge(20);y.loc[mask,'Q']*=.001;y.loc[mask,'k_exp']=-np.arange(mask.sum())-1
    for f in FIXED:y.loc[mask,f]=99999
    y,features=information(y,'prefix_fixed');b=predict_neural(X,m,payload['sx'],payload['sy'],y,features,payload['cfg']['L'])
    assert np.array_equal(a.Q_pred.to_numpy(),b.Q_pred.to_numpy())
    j.update(job_id='smoke_history',information='history_only');row2,p2=fit_job(j)
    assert row2['val_Q_MAPE_equal_cell']>0 and not p2.Q_pred.isna().any()
    receipt=dict(passed=True,fixed_future_Q_k_descriptors_tamper_invariant=True,history_only_fit=True,raw_capacity_pass=json.loads((ROOT/'audit/raw_capacity_receipt.json').read_text())['passed'],scored_test_cells=a.cell_id.nunique())
    (ROOT/'audit/smoke_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('NESTED_SMOKE_PASS',flush=True)

def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a=a.parse_args()
    plan=prepare();assert json.loads((ROOT/'audit/raw_capacity_receipt.json').read_text())['passed']
    if a.smoke:smoke();return
    assert json.loads((ROOT/'audit/smoke_receipt.json').read_text())['passed']
    (ROOT/'audit/run_environment.json').write_text(json.dumps(dict(python=sys.version,os=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),plan_sha256=sha(ROOT/'plan.json'),code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'study.py',ROOT/'run_nested_search.py',ROOT/'trainer/exp_lib.py']}),indent=2),encoding='utf8')
    run_stage(search_jobs(plan),'search');w=select_winners();run_stage(final_jobs(w),'final')
    print('NESTED_SEARCH_AND_FINAL_COMPLETE',flush=True)
if __name__=='__main__':main()
