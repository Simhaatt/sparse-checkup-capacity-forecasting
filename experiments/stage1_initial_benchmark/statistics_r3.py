"""Declared cell-level aggregation, bootstrap and explicit multiplicity families."""
from r3 import *
from scipy.stats import wilcoxon

def bootstrap_ci(values):
    x=np.asarray(values,float);assert np.isfinite(x).all() and len(x)>0
    return np.percentile(np.random.default_rng(20261005).choice(x,(10000,len(x))).mean(1),[2.5,97.5])
def holm(values):
    p=np.asarray(values,float);order=np.argsort(p);out=np.empty(len(p));largest=0.
    for k,i in enumerate(order):largest=max(largest,(len(p)-k)*p[i]);out[i]=min(1.,largest)
    return out
def cell_errors(pred):
    p=pred.copy();p['APE']=100*abs(p.Q_pred-p.Q_true)/p.Q_true
    return p.groupby(['method','origin','seed','cell_id']).APE.mean().groupby(['method','origin','cell_id']).mean().reset_index()
def summarize(cell,experiment):
    rows=[]
    for (method,origin),g in cell.groupby(['method','origin']):
        lo,hi=bootstrap_ci(g.APE);rows.append(dict(method=method,origin=origin,experiment=experiment,status='done',n_cells=len(g),Q_MAPE_equal_cell=g.APE.mean(),cell_sd=g.APE.std(),ci95_low=lo,ci95_high=hi))
    return pd.DataFrame(rows)
def paired(cell,comparisons,family):
    """comparisons=(origin,method,reference); entire named list is one Holm family."""
    rows=[]
    for origin,method,reference in comparisons:
        c=cell[cell.origin.eq(origin)].pivot(index='cell_id',columns='method',values='APE');delta=(c[method]-c[reference]).dropna()
        lo,hi=bootstrap_ci(delta);rows.append(dict(family=family,origin=origin,comparison=method+' - '+reference,method=method,reference=reference,n_cells=len(delta),difference_pp=delta.mean(),ci95_low=lo,ci95_high=hi,p_raw=wilcoxon(delta).pvalue if np.any(delta) else 1.))
    out=pd.DataFrame(rows)
    if len(out):out['family_size']=len(out);out['p_Holm']=holm(out.p_raw)
    return out

def best_neural_table():
    p=pd.read_csv(ROOT/'results/final_pred.csv');p=p[p.variant.eq('full')&p.role.eq('validation')&p.family.isin(['compact_lstm','gru','large_lstm'])].copy();p['method']=p.family
    rows=[]
    for (fold,origin),g in p.groupby(['fold','origin']):
        score=cell_errors(g).groupby('method').APE.mean().sort_values(kind='mergesort')
        rows.append(dict(fold=fold,origin=origin,best_neural=score.index[0],inner_Q_MAPE=score.iloc[0],inner_cells=g.cell_id.nunique(),selection='Validation only; mean seedwise cell errors'))
    out=pd.DataFrame(rows);assert len(out)==15;out.to_csv(ROOT/'tables/best_neural_by_fold_origin.csv',index=False);return out

def canonical():
    needed={'final':375,'static_alias':75,'resistance':300};frames=[]
    for stage,n in needed.items():
        runs=pd.read_csv(ROOT/'results'/f'{stage}_runs.csv');p=pd.read_csv(ROOT/'results'/f'{stage}_pred.csv');assert len(runs)==n and not runs.job_id.duplicated().any()
        assert set(runs.job_id)==set(p.job_id);p['method']=p.family+'__'+p.variant
        p.loc[p.variant.eq('full'),'method']=p.family
        frames.append(p[['method','family','variant','fold','origin','role','seed','cell_id','row','Q_true','Q_pred']])
    p=pd.concat(frames,ignore_index=True)
    # Local per-cell models and pretrained forecasts can be assigned to any fold role:
    # they never used population training/validation trajectories.
    simple=[]
    for f in ['reused_arima_pred.csv','reused_fair_source_pred.csv','reused_chronos_pred.csv']:
        b=pd.read_csv(ROOT/'results'/f);b=b.rename(columns={'label':'method'});simple.append(b[['method','origin','cell_id','row','Q_true','Q_pred']])
    if (ROOT/'results/chronos_tiny_pred.csv').exists():simple.append(pd.read_csv(ROOT/'results/chronos_tiny_pred.csv')[['method','origin','cell_id','row','Q_true','Q_pred']])
    simple=pd.concat(simple,ignore_index=True);parts=pd.read_csv(ROOT/'audit/cell_partitions.csv')
    for fold in range(1,6):
        mapping=parts[parts.fold.eq(fold)].set_index('cell_id').role;b=simple.copy();b['role']=b.cell_id.map(mapping);b=b[b.role.isin(['validation','calibration','test'])];b['fold']=fold
        b['family']=b.method;b['variant']='baseline'
        for seed in SEEDS:frames.append(b.assign(seed=seed))
    semi=pd.read_csv(ROOT/'results/semiempirical_pred_v3.csv');semi['family']='semiempirical';semi['variant']='population';frames.append(semi[p.columns])
    p=pd.concat(frames,ignore_index=True);keys=['method','fold','origin','role','seed','cell_id','row']
    assert not p.duplicated(keys).any();d=data();q0=d.groupby('cell_id').Q0.first();age=d.groupby('cell_id').age_type.first()
    p['Q0']=p.cell_id.map(q0);p['age_type']=p.cell_id.map(age);p['step']=p.row-p.origin+1
    assert np.isfinite(p[['Q_true','Q_pred']]).all().all() and p[['Q_true','Q_pred']].gt(0).all().all()
    return p
