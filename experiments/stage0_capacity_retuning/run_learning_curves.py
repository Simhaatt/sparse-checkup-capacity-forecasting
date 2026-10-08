"""Data-fraction-limited tuning: no reuse of full-data hyperparameter outcomes at25/50%."""
from study import *
from concurrent.futures import ProcessPoolExecutor
import run_nested_search as N

def prepare_subsets():
    p=ROOT/'audit/learning_subsets.csv'
    if p.exists():return
    rows=[]
    for fold in range(1,6):
        cells=np.array(sorted(split(fold)['fit'].cell_id.unique()));order=np.random.default_rng(2026100670+fold).permutation(cells)
        for fraction in [.25,.5,1.]:
            take=order[:max(1,int(round(len(cells)*fraction)))]
            for cell in take:rows.append(dict(fold=fold,fraction=fraction,cell_id=cell))
    pd.DataFrame(rows).to_csv(p,index=False)

def work(job):
    original=N.split;membership=pd.read_csv(ROOT/'audit/learning_subsets.csv')
    cells=set(membership[membership.fold.eq(job['fold'])&membership.fraction.eq(job['train_fraction'])].cell_id)
    def restricted(fold,d=None):
        parts=original(fold,d);parts['fit']=parts['fit'][parts['fit'].cell_id.isin(cells)].copy();return parts
    N.split=restricted
    return N.fit_job(job)

def stage(jobs,name):
    jobs=list(jobs);rp=ROOT/'results'/f'{name}_runs.csv';pp=ROOT/'results'/f'{name}_pred.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set()
    todo=[j for j in jobs if j['job_id'] not in done];print('LEARNING_STAGE',name,len(done),'/',len(jobs),flush=True)
    expected={j['job_id'] for j in jobs};assert done<=expected
    if done:
        old_runs=pd.read_csv(rp);assert not old_runs.job_id.duplicated().any()
        old_pred=pd.read_csv(pp);assert done<=set(old_pred.job_id)
        for r in old_runs.itertuples():assert sha(ROOT/'checkpoints'/f'{r.job_id}.pkl')==r.checkpoint_sha256
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for row,p in pool.map(work,todo):
            if pp.exists():
                old=pd.read_csv(pp)
                if old.job_id.eq(row['job_id']).any():old[~old.job_id.eq(row['job_id'])].to_csv(pp,index=False)
            append(pp,p);append(rp,pd.DataFrame([row]));done.add(row['job_id']);print('LEARNING_DONE',name,len(done),'/',len(jobs),row['job_id'],flush=True)
    assert done==expected

def main():
    plan=prepare();assert (ROOT/'audit/selection_receipt.json').exists();prepare_subsets()
    for fraction in [.25,.5]:
        tag=f'p{int(fraction*100)}';jobs=[]
        for j in N.search_jobs(plan):
            j=dict(j);j['job_id']='learning_'+tag+'_'+j['job_id'];j['train_fraction']=fraction;jobs.append(j)
        stage(jobs,'learning_search_'+tag)
        r=pd.read_csv(ROOT/'results'/f'learning_search_{tag}_runs.csv');winners=[];assert len(r)==500
        for (fold,family),g in r.groupby(['fold','family']):
            assert len(g)==20;w=g.sort_values(['val_Q_MAPE_equal_cell','trial']).iloc[0];winners.append(w)
        w=pd.DataFrame(winners);assert len(w)==25;w.to_csv(ROOT/'tables'/f'learning_settings_{tag}.csv',index=False);jobs=[]
        for r in w.itertuples():
            for seed in SEEDS:jobs.append(dict(job_id=f'learning_{tag}_final_f{r.fold}_{r.family}_s{seed}',stage='final',family=r.family,fold=r.fold,trial=r.trial,cfg=json.loads(r.cfg_json),seed=seed,origin=20,information='scheduled',schedule='final',train_fraction=fraction))
        stage(jobs,'learning_final_'+tag)
    (ROOT/'audit/learning_receipt.json').write_text(json.dumps(dict(passed=True,fractions=[.25,.5,1.],additional_search_fits=1000,additional_final_fits=150,reference100='Main origin20 scheduled final fits; no repeated full-data fitting.',selection='20-settings budget per family/fold/fraction, training subset only, fixed disjoint selection/calibration cells. Subsets nested and common to all families.',subset_sha256=sha(ROOT/'audit/learning_subsets.csv')),indent=2),encoding='utf8');print('LEARNING_CURVES_COMPLETE',flush=True)
if __name__=='__main__':main()
