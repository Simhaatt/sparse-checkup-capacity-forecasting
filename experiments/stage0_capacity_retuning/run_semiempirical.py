"""Fold-training population ageing law with measured-prefix cell calibration."""
from study import *
from scipy.optimize import least_squares,lsq_linear

def design(g):
    g=ordered(g);idx=np.maximum(0,np.arange(len(g))-1)
    days=(g.timestamp_s.to_numpy()-g.timestamp_s.iloc[0])/86400
    cycles=g.k_exp.to_numpy(float)
    temp=g.temperature.to_numpy(float)[idx]+273.15
    thermal=1000*(1/298.15-1/temp)
    # Source SOC describes calendar set-point; cyclic/profile SOC bounds are unavailable.
    soc=np.where(g.age_type.to_numpy()[idx]==1,g.soc_window.to_numpy(float)[idx]/100,.5)-.5
    charge=np.log1p(np.maximum(0,g.c_rate_chg.to_numpy(float)[idx]));discharge=np.log1p(np.maximum(0,g.c_rate_dischg.to_numpy(float)[idx]))
    return np.column_stack([np.sqrt(np.maximum(days[idx],0)),np.sqrt(np.maximum(cycles[idx],0)),thermal,thermal**2,soc,soc**2,charge,discharge,g.age_type.to_numpy()[idx]!=1])

def degradation(p,x):
    cal=np.exp(p[0]+p[1]*x[:,2]+p[2]*x[:,3]+p[3]*x[:,4]+p[4]*x[:,5])*x[:,0]
    cyc=np.exp(p[5]+p[6]*x[:,2]+p[7]*x[:,3]+p[8]*x[:,6]+p[9]*x[:,7])*x[:,1]*x[:,8]
    return cal+cyc

def fit_population(frame,fold):
    xx=[];yy=[];ww=[]
    for _,g0 in frame.groupby('cell_id'):
        g=ordered(g0)
        if len(g)<2:continue
        xx.append(design(g)[1:]);yy.append(1-g.Q.to_numpy(float)[1:]/g.Q0.iloc[0]);ww.extend([1/np.sqrt(len(g)-1)]*(len(g)-1))
    x=np.vstack(xx);y=np.concatenate(yy);w=np.asarray(ww)
    base=np.array([np.log(.003),0.,0.,0.,0.,np.log(.003),0.,0.,0.,0.])
    lo=np.array([-15,-20,0,-10,0,-15,-20,0,0,0.]);hi=np.array([-.5,20,20,10,20,-.5,20,20,4,4.])
    rng=np.random.default_rng(2026100650+fold);best=None;t=time.perf_counter();cpu=time.process_time()
    restarts=[]
    for restart in range(4):
        init=base.copy() if restart==0 else np.clip(base+rng.normal(0,.25,len(base)),lo+1e-8,hi-1e-8)
        r=least_squares(lambda p:(degradation(p,x)-y)*w,init,bounds=(lo,hi),max_nfev=1000,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        restarts.append(dict(restart=restart,cost=float(r.cost),success=bool(r.success),nfev=int(r.nfev)))
        if best is None or r.cost<best.cost:best=r
    assert np.isfinite(best.x).all()
    return best.x,dict(fold=fold,fit_cells=frame.cell_id.nunique(),fit_rows=len(y),fit_wall_seconds=time.perf_counter()-t,fit_cpu_seconds=time.process_time()-cpu,parameters=10,training_cost=float(best.cost),restarts=json.dumps(restarts),param_json=json.dumps(best.x.tolist()))

def forecast(params,g0,origin):
    g=ordered(g0);degrade=degradation(params,design(g));observed=g.Q.to_numpy(float)/g.Q0.iloc[0]
    # One cell-specific intercept and nonnegative ageing-rate multiplier, prefix only.
    r=lsq_linear(np.column_stack([np.ones(origin),-degrade[:origin]]),observed[:origin],bounds=([.5,0],[1.2,10]),tol=1e-10)
    pred=np.clip(r.x[0]-r.x[1]*degrade,.01,1.2)*g.Q0.iloc[0]
    return pd.DataFrame(dict(cell_id=g.cell_id.iloc[0],row=np.arange(origin,len(g)),step=np.arange(1,len(g)-origin+1),Q_true=g.Q.iloc[origin:].to_numpy(),Q_pred=pred[origin:],cell_intercept=r.x[0],cell_rate_multiplier=r.x[1]))

def main():
    prepare();out=ROOT/'results/semiempirical_pred.csv';runs=ROOT/'results/semiempirical_runs.csv';done=set(pd.read_csv(runs).fold) if runs.exists() else set()
    for fold in range(1,6):
        if fold in done:continue
        parts=split(fold);params,row=fit_population(parts['fit'],fold);rows=[]
        np.save(ROOT/'checkpoints'/f'semiempirical_f{fold}.npy',params)
        for origin in ORIGINS:
            for role in ['selection','calibration','test']:
                frame=parts[role];frame=frame[frame.cell_id.isin(cohort(data()))]
                for _,g in frame.groupby('cell_id'):
                    p=forecast(params,g,origin);p['origin']=origin;p['fold']=fold;p['role']=role;p['family']='semiempirical';p['information']='scheduled';rows.append(p)
        append(out,pd.concat(rows,ignore_index=True));append(runs,pd.DataFrame([row]));print('SEMIEMPIRICAL_FOLD',fold,'complete',flush=True)
    receipt=dict(passed=True,folds=5,parameters=10,population_fit='Fit cells only; four prespecified optimizer restarts ranked by fitting loss. No selection/calibration/test labels fitted globally.',cell_calibration='Bounded intercept[.5,1.2] and nonnegative rate multiplier[0,10] least squares on measured prefix only.',structure='Additive sqrt(elapsed_days_previous) calendar and sqrt(cycles_previous) cycle terms; exponential temperature/SOC/C-rate modifiers. U-shaped thermal terms and calendar SOC curvature.',limitations='Inspired by population calendar/cycle ageing laws; not a reproduction of Schmalstieg published coefficients. Cyclic/profile SOC bounds unavailable: nominal0.5 SOC is a disclosed placeholder, not observed SOC. Future previous-row exposure coordinates are conditioned on the benchmark schedule.',reference='https://publications.rwth-aachen.de/record/235563/; DOI10.1016/j.jpowsour.2014.02.012')
    (ROOT/'audit/semiempirical_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('SEMIEMPIRICAL_PASS',flush=True)
if __name__=='__main__':main()
