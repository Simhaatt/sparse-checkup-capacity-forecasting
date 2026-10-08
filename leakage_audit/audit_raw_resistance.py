"""Bounded remote ZIP inspection; reconstruct matched notebook processing on raw sample cells."""
from common import *
import requests,zipfile,io,ast
URL='https://www.kaggle.com/api/v1/datasets/download/simhaa/luh-blank-result-v2'
class Ranged(io.RawIOBase):
    def __init__(self):
        self.session=requests.Session();r=self.session.get(URL,stream=True,timeout=40)
        r.raise_for_status();self.url=r.url;self.size=int(r.headers['content-length']);self.pos=0;self.bytes=0;r.close()
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n;return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        if n>64*1024**2 or self.bytes+n>150*1024**2:raise ValueError('Bounded raw-data audit read limit exceeded')
        r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','Accept-Encoding':'identity'},stream=True,timeout=45)
        try:
            if r.status_code!=206:raise ValueError('Server does not support bounded byte ranges')
            data=r.raw.read(n+1)
            if len(data)!=n:raise ValueError('Incorrect range size')
            self.pos+=n;self.bytes+=n;return data
        finally:r.close()
def main():
    metadata('raw_resistance');nbpath=Path(r'C:\Users\hp\Downloads\pinn-battery-26-final.ipynb');nb=json.loads(nbpath.read_text(encoding='utf8'))
    code=''.join(nb['cells'][11]['source']);parsed=ast.parse(code)
    helpers=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in ['choose_checkup_key','extract_re_rct_from_eis']]
    namespace={'np':np,'pd':pd,'os':os};exec(compile(ast.Module(body=helpers,type_ignores=[]),str(nbpath),'exec'),namespace)
    get=namespace['extract_re_rct_from_eis'];sample=[];src=source();kept=cohort()
    for regime in [1,2,3]:sample.append(sorted(src[src.age_type.eq(regime)&src.cell_id.isin(kept)].cell_id.unique())[0])
    results=[];points=[];remote=Ranged();destination=ROOT/'audit/raw_samples';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(remote) as z:
        members=z.namelist();pd.DataFrame({'member':members}).to_csv(ROOT/'audit/public_raw_zip_members.csv',index=False)
        for cid in sample:
            candidates=[n for n in members if Path(n).name==f'cell_eisv2_{cid}.csv']
            if not candidates:raise ValueError('No raw EIS CSV member for '+cid)
            path=destination/f'cell_eisv2_{cid}.csv';path.write_bytes(z.read(candidates[0]));eis=get(str(path))
            eoc=[n for n in members if Path(n).name==f'cell_eocv2_{cid}.csv']
            raw=pd.read_csv(io.BytesIO(z.read(eoc[0])),sep=';').sort_values('timestamp_s').reset_index(drop=True)
            raw_idx=pd.to_numeric(raw.checkup_idx,errors='coerce') if 'checkup_idx' in raw else pd.Series(np.arange(len(raw)))
            g=ordered(src[src.cell_id.eq(cid)]);old=eis.rename(columns={'checkup_idx':'eis_ordinal','Re':'Re_nearest_raw'});old['eis_ordinal']=old.eis_ordinal.astype(float)
            joined=pd.merge_asof(g.sort_values('checkup_idx'),old,left_on='checkup_idx',right_on='eis_ordinal',direction='nearest')
            filled=joined.Re_nearest_raw.ffill().bfill();mu=filled.median();sd=filled.std();processed=filled.clip(mu-3*sd,mu+3*sd).cummax()
            tail=joined['checkup_idx'].isin(g.iloc[20:].checkup_idx);after=joined.checkup_idx>eis.checkup_idx.max()
            results.append(dict(cell_id=cid,age_type=int(g.age_type.iloc[0]),raw_eis_groups=len(eis),max_eis_ordinal=eis.checkup_idx.max(),max_eoc_index=g.checkup_idx.max(),
                                scored_rows=int(tail.sum()),tail_past_last_eis_ordinal=int((tail&after).sum()),join_missing_before_fill=int(joined.Re_nearest_raw.isna().sum()),
                                max_processed_Re_difference=float(abs(processed-joined.Re).max()),max_tail_raw_nearest_range=float(np.ptp(joined.loc[tail,'Re_nearest_raw'])),
                                raw_eoc_index_matches=bool(all(np.isclose(raw_idx.iloc[int(r.checkup_idx)],r.checkup_idx) and np.isclose(raw.timestamp_s.iloc[int(r.checkup_idx)],r.timestamp_s) for r in g.itertuples()))))
            joined['Re_reconstructed']=processed;joined['is_scored']=tail;joined['past_last_eis_ordinal']=after;points.append(joined[['cell_id','timestamp_s','checkup_idx','eis_ordinal','Re','Re_nearest_raw','Re_reconstructed','is_scored','past_last_eis_ordinal']])
            print('Raw resistance sample',cid,results[-1],flush=True)
    pd.DataFrame(results).to_csv(RESULTS/'raw_resistance_samples.csv',index=False);pd.concat(points).to_csv(RESULTS/'raw_resistance_alignment.csv',index=False)
    ROOT.joinpath('audit/raw_resistance_receipt.json').write_text(json.dumps({'public_dataset':URL,'range_bytes_read':remote.bytes,'notebook_sha256':hashlib.sha256(nbpath.read_bytes()).hexdigest(),'sample_cells':sample,'scope':'Three regimes sampled; all-cell tail confirmed separately; no causal attribution percentages inferred for uninspected raw cells'},indent=2))
    print('RAW_RESISTANCE_COMPLETE',flush=True)
if __name__=='__main__':main()
