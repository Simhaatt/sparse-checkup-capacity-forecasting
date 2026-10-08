"""Resumable equal-budget, cell-disjoint CPU search; never score outer test trials."""
from r3 import *
from concurrent.futures import ProcessPoolExecutor
import pickle,copy,traceback

def fit_job(job):
    import torch
    import exp_lib as X
    torch.set_num_threads(1);X.COMMON_START=job['origin']
    parts=split(job['fold']);parts={k:information(v,job['variant'])[0] for k,v in parts.items()}
    variant=job['variant'];features=F7 if variant in ['full','A'] else F6 if variant in ['no_k','static'] else [] if variant=='history_only' else F7+RE_FEATURES
    family=job['family'];cfg=copy.deepcopy(job['cfg']);cfg['L']=min(cfg['L'],job['origin'])
    tr,va=parts['fit'],parts['validation'];install_hold_adapter(X);t=time.perf_counter();cpu=time.process_time()
    path=ROOT/'checkpoints'/f"{job['job_id']}.pkl"
    if family in ['compact_lstm','gru','large_lstm']:
        fam='gru' if family=='gru' else 'lstm'
        row,_,(m,sx,sy)=X.train_eval(tr,va,None,fam,cfg,features,job['seed'],job['schedule'])
        fit_wall=time.perf_counter()-t;fit_cpu=time.process_time()-cpu
        val=predict_neural(X,m,sx,sy,va,features,cfg['L'])
        g=ordered(va[va.cell_id.eq(sorted(va.cell_id.unique())[0])]);w=sx.transform(g[features+['Q']].to_numpy(float)[job['origin']-cfg['L']:job['origin']]).astype(np.float32)
        latency=X.cpu_latency_ms(m,cfg['L'],len(features)+1,example=w)
        payload=dict(kind='neural',family=fam,cfg=cfg,features=features,information=job['variant'],state={k:v.cpu() for k,v in m.state_dict().items()},sx=sx,sy=sy,origin=job['origin'])
        cost=dict(parameters=X.count_params(m),weights_bytes=X.serialized_bytes(m),latency_cpu1_ms=latency)
    else:
        xx,yy=X.build_sequences(tr,features,cfg['L']);xx=xx.reshape(len(xx),-1);m=tree_model(family,cfg,job['seed']);m.fit(xx,yy.ravel())
        fit_wall=time.perf_counter()-t;fit_cpu=time.process_time()-cpu
        val=predict_tree(m,va,features,cfg['L'],job['origin'])
        g=ordered(va[va.cell_id.eq(sorted(va.cell_id.unique())[0])]);w=g[features+['Q']].to_numpy(float)[job['origin']-cfg['L']:job['origin']].ravel()[None]
        latency=benchmark(lambda:m.predict(w))
        payload=dict(kind='tree',family=family,cfg=cfg,features=features,information=job['variant'],model=m,origin=job['origin'])
        cost=dict(tree_nodes=sum(e.tree_.node_count for e in m.estimators_) if family=='rf' else len(m.get_booster().trees_to_dataframe()),weights_bytes=len(pickle.dumps(m)),latency_cpu1_ms=latency)
    with path.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    row={k:v for k,v in job.items() if k!='cfg'};row.update(cfg_json=json.dumps(cfg,sort_keys=True),val_Q_MAPE_equal_cell=equal_cell(val),fit_wall_seconds=fit_wall,fit_cpu_seconds=fit_cpu,serialized_bytes=path.stat().st_size,checkpoint_sha256=sha(path),train_cells=tr.cell_id.nunique(),selection_cells=va.cell_id.nunique(),trainer_sha256=sha(ROOT/'trainer/exp_lib.py'),protocol_sha256=sha(ROOT/'PLAN.json'),**cost)
    preds=[]
    roles=['validation'] if job['stage']=='search' else ['validation','calibration','test']
    for role in roles:
        frame=parts[role];frame=frame[frame.cell_id.isin(cohort(data()))]
        p=val.copy() if role=='validation' else predict_neural(X,m,sx,sy,frame,features,cfg['L']) if payload['kind']=='neural' else predict_tree(m,frame,features,cfg['L'],job['origin'])
        for k in ['job_id','family','fold','seed','origin','variant','stage']:p[k]=job[k]
        p['role']=role;preds.append(p)
    return row,pd.concat(preds,ignore_index=True)

def search_jobs(plan):
    for f in range(1,6):
        # Interleave families to give early evidence of failures/fit costs across all five.
        for trial in range(1,21):
            for family in FAMILIES:
                cfg=next(t['cfg'] for t in plan['trials'] if t['family']==family and t['trial']==trial)
                yield dict(job_id=f'search_f{f}_{family}_t{trial:02d}',stage='search',family=family,fold=f,trial=trial,cfg=cfg,seed=42,origin=20,variant='full',schedule='search')

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

def run_stage(jobs,name):
    path=ROOT/'results'/f'{name}_runs.csv';pp=ROOT/'results'/f'{name}_pred.csv'
    done=set(pd.read_csv(path).job_id) if path.exists() else set();jobs=list(jobs)
    assert done<={j['job_id'] for j in jobs}
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

def final_jobs(winners):
    for w in winners.itertuples():
        cfg=json.loads(w.cfg_json)
        for origin in [20,10,15]:
            variants=['full','no_k','history_only'] if origin==20 else ['full']
            for variant in variants:
                for seed in SEEDS:
                    yield dict(job_id=f'final_o{origin}_f{w.fold}_{w.family}_{variant}_s{seed}',stage='final',family=w.family,fold=w.fold,trial=int(w.trial),cfg=cfg,seed=seed,origin=origin,variant=variant,schedule='final')

def resistance_jobs(winners):
    for w in winners.itertuples():
        for origin in [20,10]:
            for variant in ['B','C']:
                for seed in SEEDS:
                    yield dict(job_id=f'resistance_o{origin}_f{w.fold}_{w.family}_{variant}_s{seed}',stage='resistance',family=w.family,fold=w.fold,trial=int(w.trial),cfg=json.loads(w.cfg_json),seed=seed,origin=origin,variant=variant,schedule='final')

def static_aliases():
    plan=prepare();assert plan['no_k_static_exact_input_identity']
    runs=pd.read_csv(ROOT/'results/final_runs.csv');pred=pd.read_csv(ROOT/'results/final_pred.csv')
    dest=ROOT/'results/static_alias_runs.csv';pp=ROOT/'results/static_alias_pred.csv';done=set(pd.read_csv(dest).job_id) if dest.exists() else set()
    for row in runs[runs.origin.eq(20)&runs.variant.eq('no_k')].to_dict('records'):
        source_id=row['job_id'];new_id=source_id.replace('_no_k_','_static_')
        if new_id in done:continue
        source_path=ROOT/'checkpoints'/f'{source_id}.pkl';new_path=ROOT/'checkpoints'/f'{new_id}.pkl'
        import shutil
        if not new_path.exists():shutil.copy2(source_path,new_path)
        assert sha(source_path)==sha(new_path)
        p=pred[pred.job_id.eq(source_id)].copy();p['job_id']=new_id;p['variant']='static'
        row.update(job_id=new_id,variant='static',alias_of=source_id,fit_wall_seconds=0.,fit_cpu_seconds=0.,alias_reason='All F6 descriptors constant within every source cell; exact input identity to no_k, not a statistical equivalence claim.')
        append(pp,p);append(dest,pd.DataFrame([row]));done.add(new_id)
    assert len(done)==75

def smoke():
    plan=prepare();j=next(search_jobs(plan));j.update(job_id='smoke_no_k',schedule='smoke',variant='no_k')
    row,p=fit_job(j)
    import torch,exp_lib as X
    torch.set_num_threads(1)
    payload=pickle.loads((ROOT/'checkpoints/smoke_no_k.pkl').read_bytes());m=X.make_model(payload['family'],len(payload['features'])+1,payload['cfg']);m.load_state_dict(payload['state'])
    test=split(1)['test'];test=test[test.cell_id.isin(cohort())]
    x,features=information(test,'no_k');a=predict_neural(X,m,payload['sx'],payload['sy'],x,features,payload['cfg']['L'])
    y=test.copy();mask=y.groupby('cell_id').cumcount().ge(20);y.loc[mask,'Q']=.001;y.loc[mask,'k_exp']=-99999
    y,features=information(y,'no_k');b=predict_neural(X,m,payload['sx'],payload['sy'],y,features,payload['cfg']['L'])
    assert np.array_equal(a.Q_pred,b.Q_pred)
    # C's later Re values must not enter either recursive validation or test inference.
    j.update(job_id='smoke_C',variant='C');row,p=fit_job(j);payload=pickle.loads((ROOT/'checkpoints/smoke_C.pkl').read_bytes())
    m=X.make_model(payload['family'],len(payload['features'])+1,payload['cfg']);m.load_state_dict(payload['state'])
    x,features=information(test,'C');a=predict_neural(X,m,payload['sx'],payload['sy'],x,features,payload['cfg']['L']);y=x.copy();mask=y.groupby('cell_id').cumcount().ge(20)
    y.loc[mask,['Q','Re_history','Re0_history']]=99999;y.loc[mask,'Re_mask']=1
    b=predict_neural(X,m,payload['sx'],payload['sy'],y,features,payload['cfg']['L']);assert np.array_equal(a.Q_pred,b.Q_pred)
    xx,_=X.build_sequences(split(1)['fit'],F7,j['cfg']['L']);assert xx.shape[-1]==8
    receipt=dict(passed=True,no_k_future_Q_k_tamper_invariant=True,C_future_Q_Re_Re0_mask_tamper_invariant=True,no_k_static_inputs_identical=plan['no_k_static_exact_input_identity'],target=['Q'],scored_cells=104,primary_roles='15%validation;20additional independentcalibrationcells; fitting/test disjoint',trainer_equal_cell_validation=True)
    (ROOT/'audit/smoke_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('ROUND3_SMOKE_PASS',flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();plan=prepare()
    assert json.loads((ROOT/'audit/prior_raw_capacity_receipt.json').read_text())['passed']
    if a.smoke:metadata('model_smoke');smoke();return
    assert json.loads((ROOT/'audit/smoke_receipt.json').read_text())['passed'];metadata('models')
    run_stage(search_jobs(plan),'search');w=select_winners();run_stage(final_jobs(w),'final');static_aliases();run_stage(resistance_jobs(w),'resistance')
    print('ROUND3_MODELS_COMPLETE',flush=True)
if __name__=='__main__':main()

