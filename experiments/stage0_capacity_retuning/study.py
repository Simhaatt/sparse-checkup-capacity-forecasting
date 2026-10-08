"""Fixed, cell-disjoint protocol for clean retuning and its follow-up studies."""
from pathlib import Path
import os,sys,json,hashlib,time,platform
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'trainer'))
os.environ['JES_DEVICE']='cpu';os.environ['JES_ORIGIN']='20';os.environ.pop('LEGACY_FEEDBACK',None)
F7=['k_exp','temperature','c_rate_chg','c_rate_dischg','soc_window','age_type','Q0']
FIXED=[f for f in F7 if f!='k_exp']
FAMILIES=['compact_lstm','gru','large_lstm','xgb','rf']
SEEDS=[42,52,62]
ORIGINS=[10,15,20]

def data():return pd.read_csv(ROOT/'experiments/inputs/phase1_cv_all_rows.csv')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordered(g):return g.sort_values('timestamp_s',kind='mergesort').reset_index(drop=True)
def cohort(d):return set(d.groupby('cell_id').size().loc[lambda s:s>20].index)
def append(p,df):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():df.to_csv(p,index=False);return
    h=pd.read_csv(p,nrows=0).columns.tolist()
    if set(df.columns)<=set(h):df.reindex(columns=h).to_csv(p,mode='a',header=False,index=False)
    else:pd.concat([pd.read_csv(p),df],ignore_index=True).to_csv(p,index=False)

def prepare():
    """Freeze spaces, random draws and membership before outcomes are generated."""
    plan=ROOT/'plan.json'
    if plan.exists():return json.loads(plan.read_text())
    d=data();folds=pd.read_csv(ROOT/'experiments/inputs/grouped_cv_folds.csv').set_index('cell_id').fold
    long=cohort(d);members=[]
    for f in range(1,6):
        test=set(folds[folds.eq(f)].index);eligible=np.array(sorted(long-test))
        perm=np.random.default_rng(2026100600+f).permutation(eligible)
        n=max(10,int(round(.20*len(eligible))))
        select=set(perm[:n]);cal=set(perm[n:2*n]);fit=set(d.cell_id)-test-select-cal
        assert not(fit&select or fit&cal or fit&test or select&cal or select&test or cal&test)
        for role,cells in [('fit',fit),('selection',select),('calibration',cal),('test',test)]:
            for c in sorted(cells):members.append(dict(fold=f,cell_id=c,role=role,scored=c in long))
    pd.DataFrame(members).to_csv(ROOT/'audit/cell_partitions.csv',index=False)
    spaces={
        'compact_lstm':dict(L=[8,12,16,20],hidden=[32,48,64,96],layers=[1],H=[1,3,5,8],lr='loguniform(3e-4,3e-3); require L+H<=29'),
        'gru':dict(L=[8,12,16,20],hidden=[32,64,96,128],layers=[1,2],H=[1,3,5,8],lr='loguniform(3e-4,3e-3); require L+H<=29'),
        'large_lstm':dict(L=[8,12,16,20],hidden=[128,160,192],layers=[2],H=[1,3,5,8],lr='loguniform(3e-4,3e-3); require L+H<=29'),
        'rf':dict(L=[8,12,16,20],n_estimators=[100,200,300],depth=[4,8,12,None],leaf=[1,2,4,8],max_features=[.5,.75,1.]),
        'xgb':dict(L=[8,12,16,20],n_estimators=[100,200,300],depth=[2,3,4,6],lr='loguniform(.02,.2)',subsample=[.7,.85,1.],colsample=[.7,.85,1.])}
    trials=[]
    for i,family in enumerate(FAMILIES):
        rng=np.random.default_rng(202610060+100*i);seen=set()
        while len(seen)<20:
            space=spaces[family];cfg={}
            for k,v in space.items():
                if isinstance(v,list):
                    draw=rng.choice(v);cfg[k]=draw.item() if isinstance(draw,np.generic) else draw
            # Object-valued RF depth is kept as None or integer; all draws are outcome-independent.
            if family in ['compact_lstm','gru','large_lstm']:
                cfg['lr']=float(np.exp(rng.uniform(np.log(3e-4),np.log(3e-3))))
                if cfg['L']+cfg['H']>29:continue
            elif family=='xgb':cfg['lr']=float(np.exp(rng.uniform(np.log(.02),np.log(.2))))
            key=json.dumps(cfg,sort_keys=True)
            if key in seen:continue
            seen.add(key);trials.append(dict(family=family,trial=len(seen),cfg=cfg))
    spec=dict(version=1,created_client_date='2026-10-06',origin_main=20,other_origins=[10,15],trials_per_family_per_fold=20,
        outer_folds=5,seeds=SEEDS,features=F7,ablations={'prefix_fixed':FIXED,'history_only':[]},
        fixed_descriptor_rule='Freeze all fixed descriptors to the first measured row of each cell, including in fit/selection/calibration/test. Sort only by timestamp.',
        primary_selection='Lowest equal-cell recursive Q-MAPE on selection cells; deterministic trial-index tie break. Calibration and outer test never select settings.',
        checkpoint_selection='Teacher-fed validation Q-MAPE then recursive validation Q-MAPE; selection cells only.',
        search_schedule=dict(teacher_epochs=80,teacher_patience=18,rollout_epochs=15,rollout_patience=5),
        final_schedule=dict(teacher_epochs=220,teacher_patience=40,rollout_epochs=40,rollout_patience=15),
        spaces=spaces,trials=trials,inputs={p.name:sha(p) for p in (ROOT/'experiments/inputs').glob('*.csv')},partition_sha256=sha(ROOT/'audit/cell_partitions.csv'),
        conformal='Held-out calibration cells disjoint from fit/selection/test; horizon-specific absolute-SOH residuals, finite-sample higher quantile. No finite bound if rank>n.',
        feature_ablation='Matched fold-selected structural settings, new fits/scalers/checkpoints per reduced information set; no whole-corpus feature ranking. Legacy feature ablation retired.',
        scope='20-origin clean nested random search; separately labelled10/15 origin replays of newly selected20-origin configurations, not additional random searches.')
    plan.write_text(json.dumps(spec,indent=2),encoding='utf8');return spec

def split(fold,d=None):
    d=data() if d is None else d;m=pd.read_csv(ROOT/'audit/cell_partitions.csv');m=m[m.fold.eq(fold)].set_index('cell_id').role
    return {role:d[d.cell_id.map(m).eq(role)].copy() for role in ['fit','selection','calibration','test']}
def information(d,mode):
    if mode=='scheduled':return d.copy(),F7
    if mode=='history_only':return d.copy(),[]
    x=d.sort_values(['cell_id','timestamp_s'],kind='mergesort').copy()
    for f in FIXED:x[f]=x.groupby('cell_id')[f].transform('first')
    return x,FIXED
def equal_cell(pred):return float((100*abs(pred.Q_pred-pred.Q_true)/pred.Q_true).groupby(pred.cell_id).mean().mean())
def predict_neural(X,m,sx,sy,d,features,L):
    yt,yp,meta,_=X.rollout_predict(m,sx,sy,d,features,L)
    p=meta.copy();p['Q_true']=yt[:,0];p['Q_pred']=yp[:,0];return p
def tree_model(family,cfg,seed):
    if family=='rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=cfg['n_estimators'],max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=cfg['max_features'],n_jobs=1,random_state=seed)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=cfg['n_estimators'],max_depth=cfg['depth'],learning_rate=cfg['lr'],subsample=cfg['subsample'],colsample_bytree=cfg['colsample'],n_jobs=1,random_state=seed)
def predict_tree(m,d,features,L,origin):
    rows=[];cols=features+['Q']
    for cid,g0 in d.groupby('cell_id'):
        g=ordered(g0);buf=g[cols].to_numpy(float).copy()
        for i in range(origin,len(g)):
            p=max(1e-6,float(m.predict(buf[i-L:i].ravel()[None])[0]));rows.append(dict(cell_id=cid,row=i,step=i-origin+1,age_type=int(g.age_type.iloc[i]),Q_true=g.Q.iloc[i],Q_pred=p));buf[i,-1]=p
    return pd.DataFrame(rows)
def benchmark(call):
    for _ in range(20):call()
    times=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(20):call()
        times.append((time.perf_counter()-t)*1000/20)
    return float(np.median(times))
