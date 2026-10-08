"""Post-fit numerical/figure gate, then wait for actual agent visual QA before packaging."""
from r3 import *
import subprocess

def live(pid):
    cmd=f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object -ExpandProperty CommandLine"
    r=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True);return 'continue_round3_v2.py' in r.stdout
def command(name):
    print('FINALIZATION_START',name,flush=True)
    with (ROOT/'logs'/f'final_{Path(name).stem}.log').open('a',encoding='utf8') as log:subprocess.run([sys.executable,'-X','utf8',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('FINALIZATION_DONE',name,flush=True)
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--followup-pid',type=int,required=True);a=ap.parse_args();metadata('finalization_driver')
    while live(a.followup_pid):print('WAIT_FOLLOWUP',a.followup_pid,flush=True);time.sleep(30)
    for name in ['analysis_receipt.json','extra_analysis_receipt.json','quiet_resources_receipt.json']:
        p=ROOT/'audit'/name;assert p.exists() and json.loads(p.read_text()).get('passed'), 'Follow-up process failed/incomplete; inspect before finalization.'
    command('make_plots.py');command('verify_round3.py');print('NUMERICAL_AND_FIGURES_READY; actualvisualQA required',flush=True)
    while not (ROOT/'audit/visual_qa.json').exists():time.sleep(30)
    visual=json.loads((ROOT/'audit/visual_qa.json').read_text());assert visual['passed']
    for name,digest in visual['png_hashes'].items():assert sha(ROOT/name)==digest,'Figure changed after inspection'
    command('build_complete_report.py');command('package_evidence.py');print('ROUND3_DELIVERABLES_VERIFIED',flush=True)
if __name__=='__main__':main()
