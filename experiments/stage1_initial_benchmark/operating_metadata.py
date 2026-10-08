"""Raw operating set-points for the population law; no capacity targets selected."""
from r3 import *
import zipfile,zlib,io
sys.path.insert(0,str(ROOT.parent))
from raw_zip_transport import Ranged

def nominal_soc(voltage):
    anchors=np.array([2.5,3.249,4.092,4.2]);soc=np.array([0.,.1,.9,1.]);i=int(np.argmin(abs(anchors-voltage)))
    if abs(anchors[i]-voltage)>.015:raise ValueError('Operating voltage not covered by declared published nominal-SOC anchors')
    return float(soc[i])

def main():
    metadata('operating_metadata');dest=ROOT/'audit/raw_eoc';dest.mkdir(exist_ok=True)
    out=ROOT/'results/operating_metadata.csv';done=set(pd.read_csv(out).cell_id) if out.exists() else set();source=data();remote=Ranged();manifest=[]
    with zipfile.ZipFile(remote) as archive:
        members={Path(n).name:n for n in archive.namelist()}
        for cid,g0 in source.groupby('cell_id'):
            if cid in done:continue
            filename=f'cell_eocv2_{cid}.csv';name=members[filename];info=archive.getinfo(name);path=dest/filename;cached=ROOT/'audit/raw_capacity_samples'/filename
            if not path.exists() and cached.exists():path.write_bytes(cached.read_bytes())
            if not path.exists() or (zlib.crc32(path.read_bytes())&0xffffffff)!=info.CRC:path.write_bytes(archive.read(name))
            assert (zlib.crc32(path.read_bytes())&0xffffffff)==info.CRC
            raw=pd.read_csv(path,sep=';').sort_values('timestamp_s');g=ordered(g0);age=int(g.age_type.iloc[0]);op=raw[raw.cyc_condition.eq(1)]
            charging=op[op.cyc_charged.eq(1)];discharging=op[op.cyc_charged.eq(0)]
            if age==1:
                lo=hi=float(g.soc_window.iloc[0])/100;vhigh=vlow=np.nan;available=g.timestamp_s.iloc[0]
            else:
                if not len(charging) or not len(discharging):raise ValueError('No operating charge/discharge set-point records for '+cid)
                vhigh=float(charging.v_max_target_V.iloc[0]);vlow=float(discharging.v_min_target_V.iloc[0]);hi=nominal_soc(vhigh);lo=nominal_soc(vlow);available=max(charging.timestamp_s.iloc[0],discharging.timestamp_s.iloc[0])
                if len(g)>20:assert available<=g.timestamp_s.iloc[9], 'Metadata not available by earliest requested forecast origin'
            row=dict(cell_id=cid,age_type=age,temperature=float(g.temperature.iloc[0]),calendar_or_midpoint_SOC=(hi+lo)/2,SOC_low=lo,SOC_high=hi,nominal_DoD=hi-lo,operating_v_high=vhigh,operating_v_low=vlow,metadata_observed_timestamp=available,c_rate_chg=float(g.c_rate_chg.iloc[0]),c_rate_dischg=float(g.c_rate_dischg.iloc[0]),raw_eoc_sha256=sha(path),member_crc32=info.CRC)
            append(out,pd.DataFrame([row]));done.add(cid);print('OPERATING_METADATA',len(done),'/228',cid,flush=True)
    assert len(done)==228
    receipt=dict(passed=True,cells=228,range_bytes_read_this_resume=remote.bytes,setpoints='First normal-operation(cyc_condition1) charging upper voltage and discharging lower voltage. Calendar uses prescribedstorageSOC.',nominal_SOC_map={'2.5V':0,'3.249V':.1,'4.092V':.9,'4.2V':1},FEC_definition='Nominal FEC proxy = counted operating cycles since first checkup times fixed protocol nominalDoD. Not realized futureAh throughput; no futureQ used.',limitations='Voltage-to-SOC anchors describe the published nominal protocol; ageing OCV drift is not reconstructed. Profile discharge rate0 in export denotes unavailable waveform-average rate, not0physicalcurrent.',source='Luh-Blank2024 experiment matrix, https://www.nature.com/articles/s41597-024-03831-x')
    (ROOT/'audit/operating_metadata_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('OPERATING_METADATA_COMPLETE',flush=True)
if __name__=='__main__':main()
