"""Sequential authorized follow-ups after the checkpointed primary fitting process."""
from r3 import *
import subprocess

def process_live(pid,script):
    q=f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object -ExpandProperty CommandLine"
    r=subprocess.run(['powershell','-NoProfile','-Command',q],capture_output=True,text=True);return script in r.stdout
def rows(name):
    path=ROOT/'results'/name
    try:return len(pd.read_csv(path)) if path.exists() else 0
    except pd.errors.ParserError:return -1
def run(script,*args):
    name=Path(script).stem;log=ROOT/'logs'/f'continued_{name}.log'
    print('FOLLOWUP_START',script,*args,flush=True)
    with log.open('a',encoding='utf8') as out:subprocess.run([sys.executable,'-X','utf8',script,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('FOLLOWUP_DONE',script,*args,flush=True)
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--primary-pid',type=int,required=True);a=ap.parse_args();metadata('continuation_driver')
    while process_live(a.primary_pid,'run_models.py'):
        print('WAIT_PRIMARY',a.primary_pid,'search',rows('search_runs.csv'),'final',rows('final_runs.csv'),'resistance',rows('resistance_runs.csv'),flush=True);time.sleep(30)
    expected={'search_runs.csv':500,'final_runs.csv':375,'static_alias_runs.csv':75,'resistance_runs.csv':300}
    assert all(rows(name)==n for name,n in expected.items()),'Primary process ended before complete job grid; inspect and resume it, never publish partial evidence.'
    run('analyse_round3.py');run('learning_curve.py');run('oxford_models.py');run('robustness.py','models')
    # ARIMA perturbation fits were dispatched independently and are small-memory jobs.
    while not (ROOT/'audit/robustness_arima_receipt.json').exists():
        query="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'robustness\\.py arima' } | Select-Object -ExpandProperty ProcessId"
        active=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,text=True)
        assert active.stdout.strip(),'Robust ARIMA stopped early; inspect its log and resume by jobID.'
        print('WAIT_ROBUST_ARIMA',rows('robustness_arima_runs.csv'),'/2600',flush=True);time.sleep(30)
    assert json.loads((ROOT/'audit/robustness_arima_receipt.json').read_text())['passed'];run('analyse_extras.py');run('quiet_resources.py')
    print('SECONDARY_FITTING_AND_MAIN_ANALYSIS_COMPLETE; final extra analyses, plots and completion audit still required',flush=True)
if __name__=='__main__':main()
