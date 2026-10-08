"""Independent Oxford8-cell leave-one-cell-out replication using source-selected structures."""
from study import *
from concurrent.futures import ProcessPoolExecutor
import copy,pickle
from univariate import arima_prefix,sqrt_prefix

def source_configs():
    w=pd.read_csv(ROOT/'tables/fold_selected_settings.csv');w=w[w.fold.eq(1)]
    assert len(w)==5
    return {r.family:json.loads(r.cfg_json) for r in w.itertuples()}
def oxford():return pd.read_csv(ROOT/'experiments/inputs/oxford_capacity_checkups.csv')
def oxsplit(fold):
    d=oxford();test=f'Cell{fold}';validation=f'Cell{fold%8+1}'
    d['Q']=d.Q/d.Q0
    return d[~d.cell_id.isin([test,validation])].copy(),d[d.cell_id.eq(validation)].copy(),d[d.cell_id.eq(test)].copy()
def job(j):
    import torch,exp_lib as X
    torch.set_num_threads(1);X.COMMON_START=20;tr,va,te=oxsplit(j['fold']);cfg=j['cfg'];family=j['family'];t=time.perf_counter();cpu=time.process_time()
    if family in ['compact_lstm','gru','large_lstm']:
        fam='gru' if family=='gru' else 'lstm';row,_,(m,sx,sy)=X.train_eval(tr,va,None,fam,cfg,[],j['seed'],'final')
        wall=time.perf_counter()-t;cpus=time.process_time()-cpu
        p=predict_neural(X,m,sx,sy,te,[],cfg['L']);v=predict_neural(X,m,sx,sy,va,[],cfg['L'])
        g=ordered(va);example=sx.transform(g[['Q']].to_numpy()[20-cfg['L']:20]);lat=X.cpu_latency_ms(m,cfg['L'],1,example=example)
        payload=dict(kind='neural',family=fam,cfg=cfg,features=[],state={k:v.cpu() for k,v in m.state_dict().items()},sx=sx,sy=sy,normalisation='Q/Q0',origin=20)
        cost=dict(parameters=X.count_params(m),weights_bytes=X.serialized_bytes(m),latency_cpu1_ms=lat)
    else:
        xx,yy=X.build_sequences(tr,[],cfg['L']);m=tree_model(family,cfg,j['seed']);m.fit(xx.reshape(len(xx),-1),yy.ravel());wall=time.perf_counter()-t;cpus=time.process_time()-cpu
        p=predict_tree(m,te,[],cfg['L'],20);v=predict_tree(m,va,[],cfg['L'],20);g=ordered(va);example=g[['Q']].to_numpy()[20-cfg['L']:20].ravel()[None]
        payload=dict(kind='tree',family=family,cfg=cfg,features=[],model=m,normalisation='Q/Q0',origin=20)
        cost=dict(tree_nodes=sum(e.tree_.node_count for e in m.estimators_) if family=='rf' else len(m.get_booster().trees_to_dataframe()),weights_bytes=len(pickle.dumps(m)),latency_cpu1_ms=benchmark(lambda:m.predict(example)))
    checkpoint=ROOT/'checkpoints'/f"{j['job_id']}.pkl"
    with checkpoint.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    for frame,role in [(p,'test'),(v,'selection')]:
        original=oxford().groupby('cell_id').Q0.first();frame['Q_true']*=frame.cell_id.map(original);frame['Q_pred']*=frame.cell_id.map(original);frame['role']=role
        for k in ['job_id','family','fold','seed']:frame[k]=j[k]
    row=dict(job_id=j['job_id'],family=family,fold=j['fold'],seed=j['seed'],cfg_json=json.dumps(cfg,sort_keys=True),training_cells=6,selection_cells=1,test_cells=1,val_Q_MAPE_equal_cell=equal_cell(v),fit_wall_seconds=wall,fit_cpu_seconds=cpus,serialized_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint),**cost)
    return row,pd.concat([p,v],ignore_index=True)

def controls():
    rp=ROOT/'results/oxford_controls_runs.csv';pp=ROOT/'results/oxford_controls_pred.csv';done=set(pd.read_csv(rp).cell_id) if rp.exists() else set();logs=[]
    for cid,g0 in oxford().groupby('cell_id'):
        if cid in done:continue
        g=ordered(g0);y=g.Q.to_numpy()[:20];h=len(g)-20
        a,log=arima_prefix(y,h);s=sqrt_prefix(y,h,axis=g.cycle.to_numpy());pers=np.full(h,y[-1]);frames=[]
        for method,forecast in [('ARIMA_AICc',a),('sqrt_cycle_lag1',s),('persistence',pers)]:
            frames.append(pd.DataFrame(dict(cell_id=cid,row=np.arange(20,len(g)),step=np.arange(1,h+1),Q_true=g.Q.iloc[20:].to_numpy(),Q_pred=forecast,method=method)))
        append(pp,pd.concat(frames,ignore_index=True));append(rp,pd.DataFrame([dict(cell_id=cid,**log)]));print('OXFORD_CONTROL',cid,flush=True)
    # Pinned foundation predictor on exactly the same20 measured capacities.
    if not (ROOT/'results/oxford_chronos_pred.csv').exists():
        import torch
        torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'external/python'));from chronos import BaseChronosPipeline
        pipeline=BaseChronosPipeline.from_pretrained(ROOT/'external/chronos-bolt-small',device_map='cpu',torch_dtype=torch.float32,local_files_only=True);rows=[]
        for cid,g0 in oxford().groupby('cell_id'):
            g=ordered(g0);h=len(g)-20
            with torch.no_grad():f=pipeline.predict(torch.tensor(g.Q.to_numpy()[:20],dtype=torch.float32),prediction_length=h).cpu().numpy()[0]
            median=list(pipeline.quantiles).index(.5)
            rows.append(pd.DataFrame(dict(cell_id=cid,row=np.arange(20,len(g)),step=np.arange(1,h+1),Q_true=g.Q.iloc[20:].to_numpy(),Q_pred=np.maximum(f[median,:h],1e-6),method='ChronosBolt_small_zero')))
        pd.concat(rows,ignore_index=True).to_csv(ROOT/'results/oxford_chronos_pred.csv',index=False)

def main():
    configs=source_configs();rp=ROOT/'results/oxford_runs.csv';pp=ROOT/'results/oxford_pred.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set()
    jobs=[dict(job_id=f'oxford_f{fold}_{family}_s{seed}',family=family,fold=fold,seed=seed,cfg=configs[family]) for fold in range(1,9) for family in FAMILIES for seed in SEEDS]
    assert done<={j['job_id'] for j in jobs}
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for row,pred in pool.map(job,[j for j in jobs if j['job_id'] not in done]):
            if pp.exists():
                old=pd.read_csv(pp)
                if old.job_id.eq(row['job_id']).any():old[~old.job_id.eq(row['job_id'])].to_csv(pp,index=False)
            append(pp,pred);append(rp,pd.DataFrame([row]));done.add(row['job_id']);print('OXFORD_FIT',len(done),'/120',row['job_id'],flush=True)
    assert len(done)==120;controls()
    (ROOT/'audit/oxford_models_receipt.json').write_text(json.dumps(dict(passed=True,cells=8,outer_folds=8,learned_fits=120,training_cells_per_fold=6,inner_validation='Next cyclic cell; training excludes both validation and test.',structural_choice='Only source fold1 inner-selected configurations; fixed before Oxford fitting/test. No Oxford test tuning.',features='Q/Q0 history only; scalers fitted on six training cells;3training seeds.',baselines=['ARIMA_AICc','sqrt_cycle_lag1','persistence','ChronosBolt_small_zero'],source_settings_sha256=sha(ROOT/'tables/fold_selected_settings.csv')),indent=2),encoding='utf8');print('OXFORD_MODELS_COMPLETE',flush=True)
if __name__=='__main__':main()
