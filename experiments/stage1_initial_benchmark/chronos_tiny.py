"""Additional named tiny baseline, pinned weights and native80% quantiles."""
from r3 import *
import requests
sys.path.insert(0,str(PRIOR/'external/python'))

def main():
    metadata('chronos_tiny');out=ROOT/'results/chronos_tiny_pred.csv';receipt=ROOT/'audit/chronos_tiny_receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('passed'):print('CHRONOS_TINY_ALREADY_COMPLETE');return
    pinned=ROOT/'audit/chronos_tiny_pinned_source.json'
    if pinned.exists():revision=json.loads(pinned.read_text())['revision']
    else:
        api=requests.get('https://huggingface.co/api/models/amazon/chronos-bolt-tiny',timeout=30);api.raise_for_status();record=api.json();revision=record['sha']
    folder=ROOT/'inputs/chronos-bolt-tiny';folder.mkdir(exist_ok=True)
    for name in ['config.json','model.safetensors','README.md']:
        path=folder/name
        if not path.exists():
            with requests.get(f'https://huggingface.co/amazon/chronos-bolt-tiny/resolve/{revision}/{name}',stream=True,timeout=(30,60)) as response:
                response.raise_for_status();temp=path.with_suffix(path.suffix+'.partial')
                with temp.open('wb') as f:
                    for b in response.iter_content(1024*1024):f.write(b)
                temp.replace(path)
    if pinned.exists():
        for name,digest in json.loads(pinned.read_text())['files'].items():
            assert sha(folder/name)==digest, 'Tiny weights do not match the pinned source: '+name
    import torch
    torch.set_num_threads(1);from chronos import BaseChronosPipeline
    pipeline=BaseChronosPipeline.from_pretrained(folder,device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
    done=set(map(tuple,pd.read_csv(out)[['cell_id','origin']].drop_duplicates().to_numpy())) if out.exists() else set()
    for origin in [20,10,15]:
        for cid,g0 in data().groupby('cell_id'):
            if cid not in cohort() or (cid,origin) in done:continue
            g=ordered(g0);h=len(g)-origin;context=torch.tensor(g.Q.to_numpy()[:origin],dtype=torch.float32)
            with torch.no_grad():f=pipeline.predict(context,prediction_length=h).cpu().numpy()[0]
            median=list(pipeline.quantiles).index(.5)
            p=pd.DataFrame(dict(method='ChronosBolt_tiny_zero',origin=origin,cell_id=cid,row=np.arange(origin,len(g)),Q_true=g.Q.iloc[origin:].to_numpy(),Q_pred=np.maximum(f[median,:h],1e-6),native_p10=f[0,:h],native_p90=f[-1,:h]))
            append(out,p);done.add((cid,origin))
        print('CHRONOS_TINY_ORIGIN',origin,'done',flush=True)
    assert len(done)==312
    receipt.write_text(json.dumps(dict(passed=True,model='amazon/chronos-bolt-tiny',revision=revision,cells=104,origins=[20,10,15],zero_shot=True,median_forecast=True,native_interval='p10/p90central80%, notnative90%',irregular_spacing='Order-index horizon only; elapsedphysicaltime isnotprovided.',pretraining_overlap='Notindependentlyaudited',files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}),indent=2),encoding='utf8');print('CHRONOS_TINY_COMPLETE',flush=True)
if __name__=='__main__':main()
