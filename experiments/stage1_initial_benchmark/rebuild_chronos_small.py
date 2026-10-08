"""Reproduce the reused small baseline at its actual pinned revision."""
from r3 import *
import requests
sys.path.insert(0,str(PRIOR/'external/python'))

def main():
    metadata('rebuild_chronos_small');record=json.loads((ROOT/'audit/prior_chronos_download_receipt.json').read_text());revision=record['revision'];folder=ROOT/'inputs/chronos-bolt-small';folder.mkdir(exist_ok=True)
    for name in ['config.json','model.safetensors','README.md']:
        p=folder/name
        expected=next(item['sha256'] for item in record['files'] if Path(item['file'].replace('\\','/')).name==name)
        if p.exists():
            assert sha(p)==expected, 'Cached weights do not match the pinned receipt: '+name
            continue
        with requests.get(f'https://huggingface.co/amazon/chronos-bolt-small/resolve/{revision}/{name}',stream=True,timeout=(30,60)) as r:
            r.raise_for_status();temporary=p.with_suffix(p.suffix+'.partial')
            with temporary.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
            assert sha(temporary)==expected, 'Downloaded weights do not match the pinned receipt: '+name
            temporary.replace(p)
    import torch
    torch.set_num_threads(1);from chronos import BaseChronosPipeline
    pipe=BaseChronosPipeline.from_pretrained(folder,device_map='cpu',torch_dtype=torch.float32,local_files_only=True);out=ROOT/'results/rebuilt_chronos_small_pred.csv'
    done=set(map(tuple,pd.read_csv(out)[['cell_id','origin']].drop_duplicates().to_numpy())) if out.exists() else set()
    for origin in [20,10,15]:
        groups=[(cid,ordered(g)) for cid,g in data().groupby('cell_id') if cid in cohort()]
        for start in range(0,len(groups),8):
            batch=groups[start:start+8];context=torch.tensor(np.stack([g.Q.to_numpy()[:origin] for _,g in batch]),dtype=torch.float32);h=max(len(g)-origin for _,g in batch)
            with torch.no_grad():pred=pipe.predict(context,prediction_length=h).cpu().numpy()
            mid=list(pipe.quantiles).index(.5)
            for i,(cid,g) in enumerate(batch):
                if (cid,origin) in done:continue
                n=len(g)-origin;p=pd.DataFrame(dict(cell_id=cid,origin=origin,row=np.arange(origin,len(g)),step=np.arange(1,n+1),Q_true=g.Q.iloc[origin:].to_numpy(),Q_pred=np.maximum(pred[i,mid,:n],1e-6),native_p10=pred[i,0,:n],native_p90=pred[i,-1,:n],label='ChronosBolt_small_zero'));append(out,p);done.add((cid,origin))
    assert len(done)==312
    (ROOT/'audit/rebuilt_chronos_small_receipt.json').write_text(json.dumps(dict(passed=True,revision=revision,purpose='Standalone reproduction ofreusedsmallbaseline; newoutputpathpreservesdeliveredpredictions.',cells=104,origins=[20,10,15]),indent=2),encoding='utf8');print('SMALL_BASELINE_REBUILT')
if __name__=='__main__':main()
