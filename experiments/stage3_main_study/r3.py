"""Round3 fixed protocol, cell-disjoint selection/calibration and causal input adapters."""
from pathlib import Path
import os,sys,json,hashlib,time,platform,importlib.metadata
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent.parent/'JES_capacity_retuning'
sys.path.insert(0,str(ROOT/'trainer'));os.environ['JES_DEVICE']='cpu';os.environ.pop('LEGACY_FEEDBACK',None)
F7=['k_exp','temperature','c_rate_chg','c_rate_dischg','soc_window','age_type','Q0'];F6=F7[1:]
FAMILIES=['compact_lstm','gru','large_lstm','xgb','rf'];SEEDS=[42,52,62]
RE_FEATURES=['Re_history','Re0_history','Re_mask']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data():return pd.read_csv(ROOT/'inputs/phase1_cv_all_rows.csv')
def ordered(g):return g.sort_values('timestamp_s',kind='mergesort').reset_index(drop=True)
def cohort(d=None):
    n=(data() if d is None else d).groupby('cell_id').size();return set(n[n>20].index)
def append(p,df):
    p=Path(p)
    if not p.exists():df.to_csv(p,index=False);return
    h=pd.read_csv(p,nrows=0).columns.tolist()
    if set(df.columns)<=set(h):df.reindex(columns=h).to_csv(p,mode='a',header=False,index=False)
    else:pd.concat([pd.read_csv(p),df],ignore_index=True).to_csv(p,index=False)
def metadata(name):
    out=ROOT/'audit'/f'environment_{name}.json'
    if out.exists():return
    paths=[*ROOT.glob('*.py'),*ROOT.joinpath('trainer').glob('*.py')]
    info=dict(purpose=name,client_date='2026-10-07',python=sys.version,os=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),libraries={},code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for package in ['torch','numpy','pandas','scipy','scikit-learn','xgboost','statsmodels','requests']:
        try:info['libraries'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    out.write_text(json.dumps(info,indent=2),encoding='utf8')

def prepare():
    path=ROOT/'PLAN.json'
    if path.exists():return json.loads(path.read_text())
    d=data();long=cohort(d);assert len(long)==104
    foldmap=pd.read_csv(ROOT/'inputs/grouped_cv_folds.csv').set_index('cell_id').fold;parts=[]
    for fold in range(1,6):
        test=set(foldmap[foldmap.eq(fold)].index);rest=set(d.cell_id)-test;eligible=np.array(sorted(long-test));rng=np.random.default_rng(1000+fold)
        n_val=int(round(.15*len(rest)));validation=set(rng.choice(eligible,size=n_val,replace=False))
        # Calibration is an additional held-out inner set: hyperparameter-selected residuals
        # cannot by themselves give a valid split-conformal claim.
        remaining=np.array(sorted(set(eligible)-validation));calibration=set(np.random.default_rng(2026100500+fold).choice(remaining,size=20,replace=False))
        fit=rest-validation-calibration
        for role,cells in [('fit',fit),('validation',validation),('calibration',calibration),('test',test)]:
            for cell in sorted(cells):parts.append(dict(fold=fold,cell_id=cell,role=role,scored=cell in long))
        assert not(fit&validation or fit&calibration or fit&test or validation&calibration or validation&test or calibration&test)
    pd.DataFrame(parts).to_csv(ROOT/'audit/cell_partitions.csv',index=False)
    spaces={
        'compact_lstm':dict(L=[8,12,16,20],hidden=[16,32,48,64],layers=[1],H=[1,3,5,8],lr='loguniform(3e-4,3e-3)'),
        'gru':dict(L=[8,12,16,20],hidden=[32,64,96,128],layers=[1,2],H=[1,3,5,8],lr='loguniform(3e-4,3e-3)'),
        'large_lstm':dict(L=[8,12,16,20],hidden=[128,160,192,256],layers=[1,2],H=[1,3,5,8],lr='loguniform(3e-4,3e-3)'),
        'xgb':dict(L=[8,12,16,20],n_estimators=[100,200,300],depth=[2,3,4,6],lr='loguniform(.02,.2)',subsample=[.7,.85,1.],colsample=[.7,.85,1.]),
        'rf':dict(L=[8,12,16,20],n_estimators=[100,200,300],depth=[4,8,12,None],leaf=[1,2,4,8],max_features=[.5,.75,1.])}
    trials=[]
    for fi,family in enumerate(FAMILIES):
        rng=np.random.default_rng(20261005+fi*100);seen=set()
        while len(seen)<20:
            cfg={}
            for k,v in spaces[family].items():
                if isinstance(v,list):x=rng.choice(v);cfg[k]=x.item() if isinstance(x,np.generic) else x
            if family in ['compact_lstm','gru','large_lstm']:
                if cfg['L']+cfg['H']>29:continue
                cfg['lr']=float(np.exp(rng.uniform(np.log(3e-4),np.log(3e-3))))
            elif family=='xgb':cfg['lr']=float(np.exp(rng.uniform(np.log(.02),np.log(.2))))
            key=json.dumps(cfg,sort_keys=True)
            if key in seen:continue
            seen.add(key);trials.append(dict(family=family,trial=len(seen),cfg=cfg))
    static_identical=all(d.groupby('cell_id')[f].nunique().eq(1).all() for f in F6)
    spec=dict(primary_origin=20,source_scored_cells=104,outer_folds=5,validation_fraction_of_outer_train=.15,validation_seed='1000+fold',calibration='Additional20held-out long cells/fold, disjoint from fitting and15%model-selection validation; seed2026100500+fold.',seeds=SEEDS,
        spaces=spaces,trials=trials,budget=20,search_fits=500,search_seed=42,selection='Equal-cell recursive validation Q-MAPE; earliest trial breaks ties.',
        neural_schedule={'search':[80,18,15,5],'final':[220,40,40,15]},checkpoint_metric='Equal-cell validation for teacher/recursive stages; targetQonly; correct output-to-input feedback coordinates.',
        ablations=['full','no_k','static','history_only'],no_k_static_exact_input_identity=bool(static_identical),
        ablation_selection='Only prespecified variants. Same fold-selected configuration/seeds; fitting/scalers/checkpoints inside training folds. No global feature ranking.',
        seed_aggregation='Primary: meanAPEoversteps withinseed/cell, meanoverseeds withincell, equalthenovercells. Ensemble means computedperseed, not seed-ensemble predictions.',
        inference='Rowsi-L..i-1; currentdescriptorrowexcluded; afteroriginQself-fed. Refeaturesinputonly: afteroriginholdlastpreoriginavailableRe and Re0, mask0. The same hold applies to every simulated training rollout segment.',
        resistance='B/Csameinputdimensions and trainer; BhasoldRe/Re0 withprefixmask1. Cusesbackward14dayrawalignment, missing numerical sentinel0+mask0; rawlabelsremainNaN. No futurefirst-Re0 exposure.',
        conformal='Absolute capacity residuals(Ah), independent calibration, perhorizon/method/seed/fold; quantile rankceil((n+1)*nominal). Infinite ifinsufficient n. Report80/90empirical overall/type coverage andwidth.',
        bootstrap=dict(draws=10000,seed=20261005),paired_test='Cell-paired Wilcoxon withHolmwithinexplicitnamedfamilies; no non-significanceequivalence.',
        second_source='Oxford8cell independent additionalstudy retained fromearlierrequest; inherentlydifferentcohort, reported separately from104sourcecellstudy.',
        robustness='Prefix-noise/missing-checkup additionalstudy retained fromearlierrequest; truthsafteroriginunaltered.',
        learning='TunedXGBandbestneural chosenoninner validation perouterfold.25/50/100fitting-cell subsets,3randomnested-subsetrepeats,3trainingseeds. Conditionalonfold-tunedstructures; validation/calibrationcellcounts separatelyreported.',
        inputs={p.name:sha(p) for p in (ROOT/'inputs').glob('*.csv')},partition_sha256=sha(ROOT/'audit/cell_partitions.csv'))
    path.write_text(json.dumps(spec,indent=2),encoding='utf8');return spec

def split(fold,d=None):
    d=data() if d is None else d;m=pd.read_csv(ROOT/'audit/cell_partitions.csv');m=m[m.fold.eq(fold)].set_index('cell_id').role
    return {role:d[d.cell_id.map(m).eq(role)].copy() for role in ['fit','validation','calibration','test']}
def information(d,variant):
    d=d.copy()
    if variant in ['full','A']:return d,F7
    if variant=='history_only':return d,[]
    if variant=='no_k':return d,F6
    if variant=='static':
        for f in F6:d[f]=d.groupby('cell_id')[f].transform('first')
        return d,F6
    if variant=='B':
        d['Re_history']=d.Re;d['Re0_history']=d.Re0;d['Re_mask']=1
    elif variant=='C':
        re=pd.read_csv(ROOT/'results/aligned_re_v2.csv')
        d=d.merge(re[['cell_id','checkup_idx','Re_aligned','Re_mask','Re0_available']],on=['cell_id','checkup_idx'],validate='one_to_one')
        d['Re_history']=d.Re_aligned.fillna(0.);d['Re0_history']=d.Re0_available
    else:raise ValueError(variant)
    return d,F7+RE_FEATURES

def hold_resistance(d,origin):
    if 'Re_history' not in d:return d
    rows=[]
    for _,g0 in d.groupby('cell_id',sort=False):
        g=ordered(g0).copy();valid=np.flatnonzero(g.Re_mask.to_numpy()[:origin]>0);last=float(g.Re_history.iloc[valid[-1]]) if len(valid) else 0.;anchor=float(g.Re0_history.iloc[origin-1]) if len(g)>=origin else 0.
        g.loc[origin:,'Re_history']=last;g.loc[origin:,'Re0_history']=anchor;g.loc[origin:,'Re_mask']=0;rows.append(g)
    return pd.concat(rows,ignore_index=True)
def install_hold_adapter(X):
    if getattr(X,'_round3_hold_adapter_installed',False):return
    original_predict=X.rollout_predict;original_segments=X.rollout_segments
    def predict(model,sx,sy,frame,features,L,teacher=False):
        return original_predict(model,sx,sy,hold_resistance(frame,X.COMMON_START) if 'Re_history' in features else frame,features,L,teacher)
    def segments(df,features,L,H,sx,sy,stride=2):
        x0,ex,y=original_segments(df,features,L,H,sx,sy,stride)
        if 'Re_history' not in features:return x0,ex,y
        count=0
        for _,g0 in df.groupby('cell_id',sort=False):
            g=ordered(g0)
            for i in range(L,len(g)-H+1,stride):
                valid=np.flatnonzero(g.Re_mask.to_numpy()[:i]>0);last=float(g.Re_history.iloc[valid[-1]]) if len(valid) else 0.
                for name,value in [('Re_history',last),('Re0_history',float(g.Re0_history.iloc[i-1])),('Re_mask',0.)]:
                    f=features.index(name);ex[count,:,f]=(value-sx.mean_[f])/sx.scale_[f]
                count+=1
        assert count==len(ex);return x0,ex,y
    X.rollout_predict=predict;X.rollout_segments=segments;X._round3_hold_adapter_installed=True
def equal_cell(p):return float((100*abs(p.Q_pred-p.Q_true)/p.Q_true).groupby(p.cell_id).mean().mean())
def tree_model(family,cfg,seed):
    if family=='rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=cfg['n_estimators'],max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=cfg['max_features'],n_jobs=1,random_state=seed)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=cfg['n_estimators'],max_depth=cfg['depth'],learning_rate=cfg['lr'],subsample=cfg['subsample'],colsample_bytree=cfg['colsample'],n_jobs=1,random_state=seed)
def predict_tree(m,d,features,L,origin):
    rows=[];cols=features+['Q'];d=hold_resistance(d,origin) if 'Re_history' in features else d
    for cid,g0 in d.groupby('cell_id'):
        g=ordered(g0);buf=g[cols].to_numpy(float).copy()
        for i in range(origin,len(g)):
            p=max(1e-6,float(m.predict(buf[i-L:i].ravel()[None])[0]));rows.append(dict(cell_id=cid,row=i,step=i-origin+1,age_type=int(g.age_type.iloc[i]),Q_true=g.Q.iloc[i],Q_pred=p));buf[i,-1]=p
    return pd.DataFrame(rows)
def predict_neural(X,m,sx,sy,d,features,L):
    yt,yp,meta,_=X.rollout_predict(m,sx,sy,d,features,L);p=meta.copy();p['Q_true']=yt[:,0];p['Q_pred']=yp[:,0];return p
def benchmark(call):
    for _ in range(20):call()
    ts=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(20):call()
        ts.append((time.perf_counter()-t)*1000/20)
    return float(np.median(ts))
