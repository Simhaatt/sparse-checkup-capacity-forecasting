"""Three random training-subset repeats for tuned XGB and validation-selected neural."""
from r3 import *
from concurrent.futures import ProcessPoolExecutor
import run_models as M
from statistics_r3 import best_neural_table,bootstrap_ci

def subsets():
    path=ROOT/'audit/learning_subsets.csv'
    if path.exists():return pd.read_csv(path)
    rows=[]
    for fold in range(1,6):
        cells=np.array(sorted(split(fold)['fit'].cell_id.unique()))
        for repeat in range(3):
            order=np.random.default_rng(20261005+100*fold+1000*repeat).permutation(cells)
            for fraction in [.25,.5,1.]:
                for c in order[:max(1,int(round(len(cells)*fraction)))]:rows.append(dict(fold=fold,repeat=repeat,fraction=fraction,cell_id=c))
    out=pd.DataFrame(rows);out.to_csv(path,index=False);return out

def work(j):
    original=M.split;membership=pd.read_csv(ROOT/'audit/learning_subsets.csv')
    cells=set(membership[membership.fold.eq(j['fold'])&membership.repeat.eq(j['repeat'])&membership.fraction.eq(j['fraction'])].cell_id)
    def restricted(fold,d=None):
        parts=original(fold,d);parts['fit']=parts['fit'][parts['fit'].cell_id.isin(cells)].copy();return parts
    M.split=restricted;row,p=M.fit_job(j)
    for k in ['repeat','fraction','learning_method']:p[k]=j[k]
    lengths=restricted(j['fold'])['fit'].groupby('cell_id').size();L=min(j['cfg']['L'],20);H=j['cfg'].get('H',1)
    row['teacher_examples']=int(np.maximum(lengths-L,0).sum());row['cells_contributing_examples']=int(lengths.gt(L).sum());row['rollout_segments']=int(np.maximum((lengths-L-H)//2+1,0).sum())
    return row,p

def main():
    metadata('learning_curve');subset=subsets();best=best_neural_table();best=best[best.origin.eq(20)].set_index('fold').best_neural
    winners=pd.read_csv(ROOT/'tables/fold_selected_settings.csv');jobs=[]
    for fold in range(1,6):
        for label,family in [('xgb','xgb'),('best_neural',best[fold])]:
            row=winners[winners.fold.eq(fold)&winners.family.eq(family)].iloc[0];cfg=json.loads(row.cfg_json)
            for fraction in [.25,.5]:
                for repeat in range(3):
                    for seed in SEEDS:jobs.append(dict(job_id=f'learning_p{int(fraction*100)}_f{fold}_{family}_r{repeat}_s{seed}',stage='learning',family=family,fold=fold,cfg=cfg,seed=seed,origin=20,variant='full',schedule='final',fraction=fraction,repeat=repeat,learning_method=label))
    rp=ROOT/'results/learning_runs.csv';pp=ROOT/'results/learning_pred.csv';done=set(pd.read_csv(rp).job_id) if rp.exists() else set();assert done<={j['job_id'] for j in jobs}
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for row,p in pool.map(work,[j for j in jobs if j['job_id'] not in done]):
            if pp.exists():
                old=pd.read_csv(pp)
                if old.job_id.eq(row['job_id']).any():old[~old.job_id.eq(row['job_id'])].to_csv(pp,index=False)
            append(pp,p);append(rp,pd.DataFrame([row]));done.add(row['job_id']);print('LEARNING_FIT',len(done),'/180',row['job_id'],flush=True)
    assert len(done)==180
    (ROOT/'audit/learning_receipt.json').write_text(json.dumps(dict(passed=True,new_fits=180,fractions=[.25,.5,1.],subset_repeats=3,seeds=SEEDS,methods=['xgb','best_neural'],reference100='Mainfullorigin20fits, reused; no subsamplingvariance at100%.',selection='XGB/bestneuralstructures chosen onfull-fittingpool inner-validation only. Subsets randomlydrawn insidefittingrole, shared/nestedwithinrepeat. No outerselection.',limitation='Conditionalfixed-structurelearningcurve, notcompletecampaignbudget. Full-poolHPOexposure andseparatevalidation/calibrationcellcounts mustbe disclosed.',subset_sha256=sha(ROOT/'audit/learning_subsets.csv')),indent=2),encoding='utf8');print('LEARNING_COMPLETE',flush=True)
if __name__=='__main__':main()
