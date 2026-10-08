"""Resumable byte-preserving download of the compact member of the official TAR."""
import hashlib,json,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
DEST=Path(r'F:\battery dataset\Luh_logs_v2')
URL='https://radar.kit.edu/radar-backend/archives/kww7jv8ajuvchcah/versions/1/content'

def main():
    DEST.mkdir(exist_ok=True)
    member=next(x for x in json.loads((ROOT/'exposure_audit/luh_remote_tar_inventory.json').read_text())
                if x['name'].endswith('/cell_log_age_ultracompr.7z'))
    name=Path(member['name']).name
    final=DEST/name
    part=DEST/(name+'.part')
    manifest=Path(r'F:\battery dataset\reference_documents\luh_remote_manifest-md5.txt').read_text()
    expected=next(line.split()[0] for line in manifest.splitlines() if line.endswith('dataset/'+name))
    if not final.exists():
        position=part.stat().st_size if part.exists() else 0
        assert position<=member['bytes']
        with requests.Session() as session,part.open('ab') as out:
            while position<member['bytes']:
                # Bound each response to limit buffering under memory pressure.
                # Interrupted streams remain resumable from the actual disk size.
                length=min(32*1024**2,member['bytes']-position)
                start=member['data_offset']+position
                with session.get(URL,headers={'Range':f'bytes={start}-{start+length-1}',
                                             'Accept-Encoding':'identity'},stream=True,timeout=(30,90)) as r:
                    r.raise_for_status()
                    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{start+length-1}/81909759488',r.headers.get('Content-Range')
                    received=0
                    for block in r.iter_content(1024**2):
                        if received+len(block)>length:raise ValueError('Oversized range')
                        out.write(block);received+=len(block)
                        if received%(64*1024**2)==0:
                            out.flush()
                            print(f'{position+received}/{member["bytes"]} bytes ({100*(position+received)/member["bytes"]:.1f}%)',flush=True)
                    out.flush()
                    position+=received
                    print(f'{position}/{member["bytes"]} bytes ({100*position/member["bytes"]:.1f}%)',flush=True)
                    if received!=length:raise ValueError('Incomplete range; rerun to resume')
                time.sleep(3)
        target=part
    else:target=final
    assert target.stat().st_size==member['bytes']
    md5=hashlib.md5();sha=hashlib.sha256()
    with target.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):md5.update(block);sha.update(block)
    if md5.hexdigest()!=expected:raise ValueError('Official MD5 mismatch; do not use partial file')
    if target==part:part.rename(final)
    (DEST/'download_receipt.json').write_text(json.dumps(dict(url=URL,member=member,path=str(final),
        md5=md5.hexdigest(),sha256=sha.hexdigest(),official_md5_verified=True),indent=2))
    print('Verified official MD5; complete:',final,flush=True)

if __name__=='__main__':main()
