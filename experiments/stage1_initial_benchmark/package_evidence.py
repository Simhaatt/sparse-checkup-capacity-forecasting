"""Verified evidence/source bundle; trained checkpoints remain separately hashed locally."""
from r3 import *
import zipfile

def main():
    assert json.loads((ROOT/'audit/completion_verification.json').read_text())['passed']
    assert json.loads((ROOT/'audit/visual_qa.json').read_text())['passed']
    assert (ROOT/'REPORT_COMPLETE.md').exists()
    target=ROOT.parent.parent/'JES_round3_verified_evidence.zip';files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'checkpoints' in p.parts or 'logs' in p.parts:continue
        if p.suffix in ['.zip','.pyc','.partial'] or p.name=='package_receipt.json':continue
        files.append(p)
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('BUNDLE_MANIFEST.csv',pd.DataFrame(records).to_csv(index=False))
        z.writestr('BUNDLE_README.txt','Read REPORT_COMPLETE.md and RESULTS_SUMMARY.csv first.\n'
            'This bundle includes code, inputs, raw/source receipts, predictions, tables and figures.\n'
            'Trained checkpoint payloads are excluded to keep the evidence bundle manageable; their hashes remain in runs/audit files.\n'
            'Use prepare_reproduction.py --output NEW_DIRECTORY to create a fresh fitting directory without accepted-job markers or checkpoints. Independent per-cell baselines and raw audit evidence are explicitly reused. Follow the report commands there.\n'
            'Pinned Chronos small weights/runtime originally cached at JES_capacity_retuning/external; obtain the exact model revision in audit/prior_chronos_download_receipt.json and install the recorded chronos library if that cache is absent.\n'
            'Native tiny weights are included under inputs/chronos-bolt-tiny.\n'
            'Relative report links resolve inside the extracted folder. Preserve prior evidence and separate reruns.\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for record in records:assert hashlib.sha256(z.read(record['path'])).hexdigest()==record['sha256']
    receipt=dict(passed=True,archive=str(target),members=len(records)+2,bytes=target.stat().st_size,sha256=sha(target),checkpoint_policy='Notincluded; trainedpayloadhashes retained; refitting scripts/inputsource supplied.',excluded='Logs, pycache, incomplete temporarydownloads, trainedcheckpoints.')
    (ROOT/'audit/package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print('EVIDENCE_BUNDLE_VERIFIED',target,flush=True)
if __name__=='__main__':main()
