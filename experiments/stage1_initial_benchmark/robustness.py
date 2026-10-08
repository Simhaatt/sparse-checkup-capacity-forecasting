"""Shared prefix perturbations; no future target corruption or oracle ARIMA order."""
from r3 import *
from univariate import arima_prefix,sqrt_prefix
from concurrent.futures import ProcessPoolExecutor
import pickle

def prepare_perturbations():
    path=ROOT/'results/robustness_prefixes.csv'
    if path.exists():return pd.read_csv(path)
    rows=[];jobs=[]
    for cid,g0 in data().groupby('cell_id'):
        if cid not in cohort():continue
        g=ordered(g0);original=g.Q.to_numpy()[:20];q0=float(g.Q0.iloc[0]);cell_seed=int(hashlib.sha256(cid.encode()).hexdigest()[:8],16)
        for noise_seed in SEEDS:
            rng=np.random.default_rng(cell_seed+noise_seed);normal=rng.standard_normal(20);normal[0]=0;order=rng.permutation(np.arange(1,19))
            for sigma in [0.,.005,.01]:
                for drop in [0.,.1,.25]:
                    # One clean baseline suffices; randomized perturbations have three repeats.
                    if sigma==drop==0 and noise_seed!=42:continue
                    missing=order[:int(round(20*drop))];observed=np.maximum(original+sigma*q0*normal,1e-6);observed[missing]=np.nan;imputed=pd.Series(observed).ffill().to_numpy()
                    job_id=f'robust_{cid}_n{int(sigma*10000)}_d{int(drop*100)}_r{noise_seed}'
                    assert np.isfinite(observed[[0,19]]).all() and np.isfinite(imputed).all()
                    jobs.append(dict(job_id=job_id,cell_id=cid,noise_sigma_Q0=sigma,drop_fraction=drop,noise_seed=noise_seed,observed_checkups=int(np.isfinite(observed).sum())))
                    for row in range(20):rows.append(dict(job_id=job_id,cell_id=cid,row=row,Q_original=original[row],Q_observed=observed[row],Q_imputed=imputed[row],observed=np.isfinite(observed[row])))
    out=pd.DataFrame(rows);out.to_csv(path,index=False);pd.DataFrame(jobs).to_csv(ROOT/'audit/robustness_jobs.csv',index=False)
    plan=dict(origin=20,source_cells=104,noise_sigmas_Q0=[0,.005,.01],drop_fractions=[0,.1,.25],noise_seeds=SEEDS,initial_reference_and_last_checkup_preserved=True,shared_nested_noise_and_dropout_across_models=True,clean_repetitions=1,noisy_repetitions=3,unique_scenarios_per_cell=25,truth_after_origin='Never modified',interpretation='Synthetic measurement/missed-checkup sensitivity, not an estimated laboratory-noise model or validated field distribution. Initialcapacity reference preserved.',Q_missing_policy='Learned causal last-observation carry-forward; ARIMA native NaNs on nominal index grid; square-root fits use observed prefix positions only.')
    (ROOT/'audit/robustness_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8');return out

def arima_job(j):
    prefixes=pd.read_csv(ROOT/'results/robustness_prefixes.csv');p=prefixes[prefixes.job_id.eq(j['job_id'])].sort_values('row');g=ordered(data()[data().cell_id.eq(j['cell_id'])]);h=len(g)-20
    forecast,log=arima_prefix(p.Q_observed.to_numpy(),h);control=sqrt_prefix(p.Q_observed.to_numpy(),h);pers=np.repeat(p.Q_observed.iloc[-1],h);frames=[]
    for method,values in [('ARIMA_AICc',forecast),('sqrt_idx_lag1',control),('persistence',pers)]:
        q=pd.DataFrame(dict(cell_id=j['cell_id'],row=np.arange(20,len(g)),Q_true=g.Q.iloc[20:].to_numpy(),Q_pred=values,method=method))
        for k,v in j.items():q[k]=v
        q['model_seed']=0;frames.append(q)
    return {**j,**log},pd.concat(frames,ignore_index=True)

def run_arima():
    metadata('robustness_arima_manual_batches');prepare_perturbations();jobs=pd.read_csv(ROOT/'audit/robustness_jobs.csv').to_dict('records');rp=ROOT/'results/robustness_arima_runs.csv';pp=ROOT/'results/robustness_arima_pred.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set()
    pending=[j for j in jobs if j['job_id'] not in done]
    for start in range(0,len(pending),25):
        with ProcessPoolExecutor(max_workers=1) as pool:
            for row,p in pool.map(arima_job,pending[start:start+25]):
                if pp.exists():
                    old=pd.read_csv(pp)
                    if old.job_id.eq(row['job_id']).any():old[~old.job_id.eq(row['job_id'])].to_csv(pp,index=False)
                append(pp,p);append(rp,pd.DataFrame([row]));done.add(row['job_id']);print('ROBUST_ARIMA',len(done),'/2600',row['job_id'],row['status'],flush=True)
    assert len(done)==2600
    (ROOT/'audit/robustness_arima_receipt.json').write_text(json.dumps(dict(passed=True,jobs=2600,source_cells=104,scenarios_per_cell=25,selection='AICc rebuilt on actually observed perturbed prefix, no clean-oracle orders.',missing='Native state-space missing observations; effective AICc count uses observedprefix count minus differencing.',fallbacks_logged=True),indent=2),encoding='utf8');print('ROBUST_ARIMA_COMPLETE',flush=True)

def model_job(j):
    import torch,exp_lib as X
    torch.set_num_threads(1);X.COMMON_START=20
    path=ROOT/'checkpoints'/f"final_o20_f{j['fold']}_{j['family']}_full_s{j['model_seed']}.pkl";payload=pickle.loads(path.read_bytes())
    if payload['kind']=='neural':
        model=X.make_model(payload['family'],len(payload['features'])+1,payload['cfg']);model.load_state_dict(payload['state']);model.eval()
    else:model=payload['model']
    parts=split(j['fold']);test=parts['test'][parts['test'].cell_id.isin(cohort())]
    prefixes=pd.read_csv(ROOT/'results/robustness_prefixes.csv');jobs=pd.read_csv(ROOT/'audit/robustness_jobs.csv');frames=[]
    for job in jobs[jobs.cell_id.isin(set(test.cell_id))].to_dict('records'):
        g=ordered(test[test.cell_id.eq(job['cell_id'])]).copy();x=prefixes[prefixes.job_id.eq(job['job_id'])].sort_values('row');g.loc[:19,'Q']=x.Q_imputed.to_numpy()
        pred=predict_neural(X,model,payload['sx'],payload['sy'],g,payload['features'],payload['cfg']['L']) if payload['kind']=='neural' else predict_tree(model,g,payload['features'],payload['cfg']['L'],20)
        for k,v in job.items():pred[k]=v
        pred['method']=j['family'];pred['model_seed']=j['model_seed'];pred['fold']=j['fold'];pred['model_job_id']=j['job_id'];frames.append(pred)
    return dict(**j,checkpoint_sha256=sha(path),predicted_scenarios=len(frames)),pd.concat(frames,ignore_index=True)

def run_models():
    metadata('robustness_models');prepare_perturbations();jobs=[dict(job_id=f'robust_model_f{fold}_{family}_s{seed}',fold=fold,family=family,model_seed=seed) for fold in range(1,6) for family in FAMILIES for seed in SEEDS]
    rp=ROOT/'results/robustness_model_runs.csv';pp=ROOT/'results/robustness_model_pred.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set()
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for row,p in pool.map(model_job,[j for j in jobs if j['job_id'] not in done]):
            if pp.exists():
                old=pd.read_csv(pp)
                if old.model_job_id.eq(row['job_id']).any():old[~old.model_job_id.eq(row['job_id'])].to_csv(pp,index=False)
            append(pp,p);append(rp,pd.DataFrame([row]));done.add(row['job_id']);print('ROBUST_MODEL',len(done),'/75',row['job_id'],flush=True)
    assert len(done)==75
    (ROOT/'audit/robustness_models_receipt.json').write_text(json.dumps(dict(passed=True,model_jobs=75,source_cells=104,scenarios_per_cell=25,training='No retraining on noisyoutercells; evaluates fixedfold-tuned modelresponse.',future_Q_truth_unchanged=True),indent=2),encoding='utf8');print('ROBUST_MODELS_COMPLETE',flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','arima','models']);a=ap.parse_args()
    if a.stage=='prepare':metadata('robustness_prepare');prepare_perturbations();print('ROBUSTNESS_PLAN_READY')
    elif a.stage=='arima':run_arima()
    else:run_models()
if __name__=='__main__':main()
