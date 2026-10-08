"""Pinned Chronos-Bolt small CPU zero-shot forecasts, with no fine-tuning/test selection."""
from study import *
sys.path.insert(0,str(ROOT/'external/python'))
import importlib.metadata

def main():
    import torch
    torch.set_num_threads(1)
    from chronos import BaseChronosPipeline
    model_path=ROOT/'external/chronos-bolt-small';t=time.perf_counter()
    pipeline=BaseChronosPipeline.from_pretrained(model_path,device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
    load_seconds=time.perf_counter()-t;d=data();long=cohort(d);rows=[];costs=[]
    for origin in ORIGINS:
        groups=[(cid,ordered(g)) for cid,g in d.groupby('cell_id') if cid in long]
        t=time.perf_counter();cpu=time.process_time()
        for start in range(0,len(groups),8):
            batch=groups[start:start+8];context=torch.tensor(np.stack([g.Q.to_numpy(float)[:origin] for _,g in batch]),dtype=torch.float32)
            horizon=max(len(g)-origin for _,g in batch)
            with torch.no_grad():forecast=pipeline.predict(context,prediction_length=horizon).cpu().numpy()
            median_index=list(pipeline.quantiles).index(.5)
            for j,(cid,g) in enumerate(batch):
                for step in range(len(g)-origin):rows.append(dict(cell_id=cid,origin=origin,row=origin+step,step=step+1,Q_true=g.Q.iloc[origin+step],Q_pred=max(1e-6,float(forecast[j,median_index,step])),native_p10=float(forecast[j,0,step]),native_p90=float(forecast[j,-1,step]),label='ChronosBolt_small_zero'))
        costs.append(dict(origin=origin,cells=len(groups),inference_wall_seconds=time.perf_counter()-t,inference_cpu_seconds=time.process_time()-cpu,fit_cpu_seconds=0.))
        print('CHRONOS_ORIGIN',origin,'complete',flush=True)
    p=pd.DataFrame(rows);p.to_csv(ROOT/'results/chronos_pred.csv',index=False);pd.DataFrame(costs).to_csv(ROOT/'tables/chronos_cost.csv',index=False)
    assert p.groupby('origin').cell_id.nunique().eq(104).all() and np.isfinite(p.Q_pred).all()
    download=json.loads((ROOT/'audit/chronos_download_receipt.json').read_text())
    receipt=dict(passed=True,model=download['model_id'],revision=download['revision'],zero_shot=True,library_version=importlib.metadata.version('chronos-forecasting'),torch_version=torch.__version__,device='cpu',threads=1,parameters=sum(x.numel() for x in pipeline.model.parameters()),load_seconds=load_seconds,weight_file_bytes=(model_path/'model.safetensors').stat().st_size,contexts='Only measured Q prefixes; index-time horizons like ARIMA; no exogenous variables or fine-tuning.',point_estimator='Native0.5 quantile; positive clipping only; no cumulative-minimum postprocessing.',pretraining_overlap='No fine-tuning on this corpus; exact pretraining overlap was not independently audited.',receipt_sha256=sha(ROOT/'audit/chronos_download_receipt.json'))
    (ROOT/'audit/chronos_inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('CHRONOS_PASS',flush=True)
if __name__=='__main__':main()
