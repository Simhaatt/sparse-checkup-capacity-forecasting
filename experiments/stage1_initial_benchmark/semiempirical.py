"""Declared Arrhenius/power population model and one prefix-only cell scale."""
from r3 import *
from scipy.optimize import least_squares
NAMES=['log_a_cal','Ea_cal_kJ_mol','SOC_cal_coef','log_a_cyc','Ea_cyc_kJ_mol','charge_rate_coef','discharge_rate_coef','DoD_coef','z_calendar','w_cycle']

def design(g,op):
    g=ordered(g);idx=np.maximum(np.arange(len(g))-1,0)
    days=(g.timestamp_s.to_numpy()-g.timestamp_s.iloc[0])/86400
    fec=g.k_exp.to_numpy(float)*float(op.nominal_DoD)
    thermal=1000/8.314462618*(1/298.15-1/(g.temperature.to_numpy()[idx]+273.15))
    return np.column_stack([days[idx],fec[idx],thermal,np.full(len(g),op.calendar_or_midpoint_SOC-.5),np.log1p(np.maximum(g.c_rate_chg.to_numpy()[idx],0)),np.log1p(np.maximum(g.c_rate_dischg.to_numpy()[idx],0)),np.full(len(g),op.nominal_DoD-.9)])
def degradation(p,x):
    cal=np.exp(p[0]+p[1]*x[:,2]+p[2]*x[:,3])*np.maximum(x[:,0],0)**p[8]
    cyc=np.exp(p[3]+p[4]*x[:,2]+p[5]*x[:,4]+p[6]*x[:,5]+p[7]*x[:,6])*np.maximum(x[:,1],0)**p[9]
    return cal+cyc
def fit(frame,ops,fold):
    xx=[];yy=[];weights=[]
    for cid,g0 in frame.groupby('cell_id'):
        g=ordered(g0)
        if len(g)<2:continue
        xx.append(design(g,ops.loc[cid])[1:]);yy.append(1-g.Q.to_numpy()[1:]/g.Q0.iloc[0]);weights.extend([1/np.sqrt(len(g)-1)]*(len(g)-1))
    x=np.vstack(xx);y=np.concatenate(yy);w=np.asarray(weights)
    initial=np.array([np.log(.002),30.,2.,np.log(.003),-10.,.2,.2,1.,.5,.5])
    lower=np.array([-18,0,0,-18,-100,0,0,0,.1,.1]);upper=np.array([0,100,10,0,100,4,4,10,1.5,1.5])
    rng=np.random.default_rng(20261005+fold);restarts=[];best=None;t=time.perf_counter();cpu=time.process_time()
    for repeat in range(4):
        init=initial.copy() if repeat==0 else np.clip(initial+rng.normal(0,.1,len(initial)),lower+1e-6,upper-1e-6)
        r=least_squares(lambda p:(degradation(p,x)-y)*w,init,bounds=(lower,upper),max_nfev=2000,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        restarts.append(dict(repeat=repeat,success=bool(r.success),cost=float(r.cost),nfev=int(r.nfev)))
        if best is None or r.cost<best.cost:best=r
    return best.x,dict(fold=fold,train_cells=frame.cell_id.nunique(),train_rows=len(y),fit_wall_seconds=time.perf_counter()-t,fit_cpu_seconds=time.process_time()-cpu,optimization_success=bool(best.success),training_cost=float(best.cost),jacobian_condition=float(np.linalg.cond(best.jac)),restarts_json=json.dumps(restarts),**dict(zip(NAMES,best.x)))
def predict(params,g0,op,origin):
    g=ordered(g0);d=degradation(params,design(g,op));loss=1-g.Q.to_numpy()[:origin]/g.Q0.iloc[0];den=float(d[:origin]@d[:origin])
    scale=float(np.clip(d[:origin]@loss/den,0,10)) if den>1e-20 else 0.
    pred=np.maximum((1-scale*d)*g.Q0.iloc[0],1e-6)
    return pd.DataFrame(dict(cell_id=g.cell_id.iloc[0],row=np.arange(origin,len(g)),step=np.arange(1,len(g)-origin+1),Q_true=g.Q.iloc[origin:].to_numpy(),Q_pred=pred[origin:],cell_scale=scale))
def main():
    metadata('semiempirical');assert json.loads((ROOT/'audit/operating_metadata_receipt.json').read_text())['passed'];ops=pd.read_csv(ROOT/'results/operating_metadata.csv').set_index('cell_id')
    rp=ROOT/'results/semiempirical_runs.csv';pp=ROOT/'results/semiempirical_pred.csv';done=set(pd.read_csv(rp).fold) if rp.exists() else set()
    for fold in range(1,6):
        if fold in done:continue
        parts=split(fold);params,row=fit(parts['fit'],ops,fold);assert np.isfinite(params).all();np.save(ROOT/'checkpoints'/f'semiempirical_f{fold}.npy',params);rows=[]
        for origin in [20,10,15]:
            for role in ['validation','calibration','test']:
                for cid,g in parts[role][parts[role].cell_id.isin(cohort())].groupby('cell_id'):
                    p=predict(params,g,ops.loc[cid],origin);p['fold']=fold;p['origin']=origin;p['role']=role;p['method']='semiempirical';p['seed']=0;rows.append(p)
        append(pp,pd.concat(rows,ignore_index=True));append(rp,pd.DataFrame([row]));print('SEMIEMPIRICAL_DONE',fold,'/5',flush=True)
    # Verify no future target enters the per-cell scale or the predicted values.
    checks=[]
    for fold in range(1,6):
        params=np.load(ROOT/'checkpoints'/f'semiempirical_f{fold}.npy');g=next(g for _,g in split(fold)['test'].groupby('cell_id') if len(g)>20);g=ordered(g);y=g.copy();y.loc[20:,'Q']=.001
        a=predict(params,g,ops.loc[g.cell_id.iloc[0]],20);b=predict(params,y,ops.loc[g.cell_id.iloc[0]],20);checks.append(np.array_equal(a.Q_pred,b.Q_pred))
    assert all(checks)
    (ROOT/'audit/semiempirical_receipt.json').write_text(json.dumps(dict(passed=True,population_fits=5,parameters=10,form='Q/Q0=1-alpha_cell*(a_cal(T,SOC)*t_previous^z+a_cyc(T,C-rate,DoD)*FEC_previous^w)',cell_calibration='Exactly one scale, bounded[0,10], least squares on the measured prefix. No fitted per-cell intercept.',global_fit='Fitting-role cells only; four optimization starts chosen by training loss, no outer labels.',future_Q_tamper_invariant=True,FEC='Nominal cycle-count-times-protocol-DoD proxy; no realized futureAh/capacity-derived exposure.',limits='Empirical population comparator, not reproduction of Schmalstieg coefficients. Ambient ageing setpoint rather than modeled coretemperature; exported profile dischargerate0 is missing waveform-average-rate information. Additional fixed SOC-window metadata beyond F7 is disclosed; no identical-information claim.',operating_metadata_sha256=sha(ROOT/'results/operating_metadata.csv')),indent=2),encoding='utf8');print('SEMIEMPIRICAL_COMPLETE',flush=True)
if __name__=='__main__':main()
