"""Prefix-only univariate controls, with missing checkups retained on the index grid."""
import itertools,warnings,time
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_prefix(prefix,horizon):
    y=np.asarray(prefix,float);observed=np.isfinite(y);center=float(np.nanmean(y));scale=float(np.nanstd(y));last=float(y[observed][-1]);start=time.perf_counter();cpu=time.process_time()
    if scale<1e-10:return np.full(horizon,last),dict(status='constant_prefix',order='-',trend='-',valid_candidates=0,fit_wall_seconds=time.perf_counter()-start,fit_cpu_seconds=time.process_time()-cpu)
    z=(y-center)/scale;best=None;valid=0;errors=0
    for p,d,q in itertools.product(range(3),range(2),range(3)):
        for trend in (['c'] if d==0 else ['n','t']):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');fit=ARIMA(z,order=(p,d,q),trend=trend).fit(method_kwargs={'maxiter':200,'disp':0})
                n=int(observed.sum())-d;k=len(fit.params);aicc=float(fit.aic)+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
                forecast=np.asarray(fit.forecast(horizon))*scale+center
                okay=bool(fit.mle_retvals.get('converged',True)) and np.isfinite(aicc) and np.isfinite(forecast).all()
                if okay:
                    valid+=1
                    if best is None or aicc<best[0]:best=(aicc,forecast,(p,d,q),trend)
            except (ValueError,np.linalg.LinAlgError,RuntimeError):errors+=1
    pred=np.full(horizon,last) if best is None else np.maximum(best[1],1e-6)
    return pred,dict(status='fallback_persistence' if best is None else 'selected',order=','.join(map(str,best[2])) if best else '-',trend=best[3] if best else '-',aicc=best[0] if best else np.nan,valid_candidates=valid,candidate_errors=errors,observed_prefix=int(observed.sum()),fit_wall_seconds=time.perf_counter()-start,fit_cpu_seconds=time.process_time()-cpu)

def sqrt_prefix(prefix,horizon,axis=None):
    y=np.asarray(prefix,float);n=len(y);axis=np.arange(n+horizon,dtype=float) if axis is None else np.asarray(axis,float)
    previous=np.maximum(np.arange(n+horizon)-1,0);x=np.sqrt(np.maximum(axis[previous],0));valid=np.isfinite(y)
    beta=np.linalg.lstsq(np.column_stack([np.ones(n),x[:n]])[valid],y[valid],rcond=None)[0]
    beta[1]=min(0.,beta[1]);return np.maximum(beta[0]+beta[1]*x[n:],1e-6)
