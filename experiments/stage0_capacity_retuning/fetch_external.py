"""Fetch identified public sources without executing remote code or altering global packages."""
from pathlib import Path
import argparse,json,requests,hashlib,time
ROOT=Path(__file__).resolve().parent

def download(url,path,expected=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (expected is None or path.stat().st_size==expected):return
    temp=path.with_suffix(path.suffix+'.partial')
    with requests.get(url,stream=True,timeout=(30,60)) as r:
        r.raise_for_status()
        with temp.open('wb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
    if expected is not None and temp.stat().st_size!=expected:raise ValueError('Source byte count differs from public record')
    temp.replace(path);print('FETCHED',path.name,path.stat().st_size,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['chronos','oxford']);args=ap.parse_args();files=[]
    if args.mode=='chronos':
        r=requests.get('https://huggingface.co/api/models/amazon/chronos-bolt-small',timeout=30);r.raise_for_status();record=r.json();revision=record['sha']
        (ROOT/'audit/chronos_model_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        for name in ['config.json','generation_config.json','model.safetensors','README.md']:
            if name not in {s['rfilename'] for s in record['siblings']}:continue
            url=f'https://huggingface.co/amazon/chronos-bolt-small/resolve/{revision}/{name}';path=ROOT/'external/chronos-bolt-small'/name
            download(url,path);files.append(dict(file=str(path.relative_to(ROOT)),url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        meta=dict(model_id='amazon/chronos-bolt-small',revision=revision,license='Apache-2.0 per official model card',files=files)
    else:
        record=json.loads((ROOT/'audit/oxford_record.json').read_text())
        for f in record['distribution']:
            if f['name'] not in ['Oxford_Battery_Degradation_Dataset_1.mat','Readme.txt']:continue
            path=ROOT/'external/oxford'/f['name'];download(f['contentUrl'],path,int(f['contentSize']))
            files.append(dict(file=str(path.relative_to(ROOT)),url=f['contentUrl'],bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        meta=dict(dataset='Oxford Battery Degradation Dataset 1',record='https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac',license=record.get('license'),files=files)
    (ROOT/'audit'/f'{args.mode}_download_receipt.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print('DOWNLOAD_COMPLETE',args.mode,flush=True)
if __name__=='__main__':main()
