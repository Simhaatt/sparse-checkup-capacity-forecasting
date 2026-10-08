"""Level-1 reproduction: re-derive the reported results from the deposited predictions, without retraining.

Steps
  1. Build a temporary copy of the main-study workspace (experiments/stage3_main_study) with the deposited inputs
     and predictions, then run the ORIGINAL analysis scripts unchanged:
        r4_analyse.py        -> primary-benchmark accuracy and the pre-declared Holm families (Table 2, Fig. 3)
        r4_decisions_v2.py   -> 80 %-capacity crossing decisions (Table D.2, Fig. 6d)
        r4_g.py g5           -> regime-stratified (Mondrian) conformal coverage (Fig. 6b)
  2. Run summarise_from_predictions.py (transfer, Oxford, split-conformal coverage, robustness, ablations,
     horizon, regime and post hoc tables).
  3. Compare every regenerated table with the deposited table in results/tables/main_study (tolerance 1e-9).
  4. Check the headline numbers quoted in the manuscript (reproduce/expected_values.json).
  5. Re-check the plotted figure values (paper_figures/scripts/check_figure_values.py).

Sandia (independent benchmark) values are checked against the deposited aggregate tables only: the Sandia
cell-level data are not redistributed (see DATA_LICENSE.md); re-deriving them needs a Level-2 run.

Usage:  python reproduce/level1.py            (writes results/reproduction_check.json; exit code 1 on failure)
"""
import json, shutil, subprocess, sys, tempfile, gzip
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STAGE3 = REPO / 'experiments' / 'stage3_main_study'
PRED = REPO / 'results' / 'predictions'
DEP = REPO / 'results' / 'tables' / 'main_study'
DATA = REPO / 'data' / 'processed'
report = {'tables': {}, 'values': [], 'figures': None}
fail = []


def ungz(src, dst):
    with gzip.open(src, 'rb') as f, open(dst, 'wb') as g: shutil.copyfileobj(f, g)


def compare(new, dep, **kw):
    a = pd.read_csv(new, **kw); b = pd.read_csv(dep, **kw)
    if a.shape != b.shape: return False, f'shape {a.shape} vs {b.shape}'
    if kw.get('index_col') is not None:
        a = a.sort_index(); b = b.sort_index(); x, y = a.to_numpy(float), b.reindex(index=a.index).to_numpy(float)
        ok = np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True); return ok, '' if ok else 'values differ'
    for c in b.columns:
        if c not in a.columns: return False, f'missing column {c}'
        if pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True):
                return False, f'column {c} differs (max {np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))):.3g})'
        elif not (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all():
            return False, f'column {c} differs'
    return True, ''


def run(cmd, cwd):
    r = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-4000:]); raise SystemExit(f'FAILED: {" ".join(cmd)}')


with tempfile.TemporaryDirectory() as tmp:
    W = Path(tmp) / 'stage3_main_study'; shutil.copytree(STAGE3, W, ignore=shutil.ignore_patterns('__pycache__'))
    for d in ('inputs', 'results', 'tables', 'audit'): (W / d).mkdir(exist_ok=True)
    shutil.copy(DATA / 'luh_blank' / 'phase1_cv_all_rows.csv', W / 'inputs')
    shutil.copy(DATA / 'luh_blank' / 'grouped_cv_folds.csv', W / 'inputs')
    shutil.copy(DATA / 'luh_blank' / 'cell_partitions.csv', W / 'audit')
    ungz(PRED / 'r3_reference_predictions.csv.gz', W / 'inputs' / 'r3_reference_predictions.csv')
    for n in ('r4_hybrid_pred', 'r4_final_pred', 'r4_hybrid_fitonly_pred'): ungz(PRED / f'{n}.csv.gz', W / 'results' / f'{n}.csv')
    shutil.copy(PRED / 'r4_final_runs.csv', W / 'results')
    print('[1/5] running the original analysis scripts ...', flush=True)
    run(['r4_analyse.py'], W); run(['r4_decisions_v2.py'], W); run(['r4_g.py', 'g5'], W)
    for n in ('r4_summary.csv', 'r4_pairs.csv', 'r4_decisions_luh_r3rule.csv', 'r4_g5_mondrian_by_regime.csv', 'r4_g5_mondrian_overall.csv'):
        ok, why = compare(W / 'tables' / n, DEP / n); report['tables'][n] = 'PASS' if ok else f'FAIL: {why}'
        if not ok: fail.append(n)
    print('[2/5] recomputing summary tables from predictions ...', flush=True)
    out = Path(tmp) / 'summaries'
    run([str(STAGE3 / 'summarise_from_predictions.py'), '--pred', str(PRED), '--data', str(DATA), '--out', str(out)], STAGE3)
    pairs = {n: {} for n in ('r4_hybrid_error_by_horizon.csv', 'r4_hybrid_vs_round3_posthoc.csv', 'r4_hybrid_vs_tuned_learned_posthoc.csv',
                             'r4_transfer_summary.csv', 'r4_transfer_pairs.csv', 'r4_g1_oxford_summary.csv', 'r4_g1_oxford_pairs.csv',
                             'r4_hybrid_conformal_luh.csv', 'r4_g4_robust_summary.csv', 'r4_g4_robust_pairs.csv',
                             'r4_ablation_luh_summary.csv', 'r4_ablation_luh_pairs.csv')}
    pairs['r4_by_age_type.csv'] = dict(header=[0, 1], index_col=0)
    print('[3/5] comparing with the deposited tables ...', flush=True)
    for n, kw in pairs.items():
        ok, why = compare(out / n, DEP / n, **kw); report['tables'][n] = 'PASS' if ok else f'FAIL: {why}'
        if not ok: fail.append(n)
    g3 = pd.read_csv(out / 'r4_g3_vs_main_pairs_luh.csv'); g3d = pd.read_csv(DEP / 'r4_g3_vs_main_pairs.csv'); g3d = g3d[g3d.dataset.eq('luh')]
    ok = np.allclose(g3[['diff_noschedule_minus_main', 'p']].to_numpy(), g3d[['diff_noschedule_minus_main', 'p']].to_numpy(), rtol=1e-9)
    report['tables']['r4_g3_vs_main_pairs.csv (Luh rows)'] = 'PASS' if ok else 'FAIL'
    if not ok: fail.append('r4_g3_vs_main_pairs')

print('[4/5] checking the headline numbers quoted in the manuscript ...', flush=True)
T = lambda n: pd.read_csv(DEP / n)
S = T('r4_summary.csv').set_index(['method', 'origin']).Q_MAPE
P = T('r4_pairs.csv').set_index(['family', 'comparison']).p_holm
for chk in json.loads((REPO / 'reproduce' / 'expected_values.json').read_text()):
    t, key, exp, dec = chk['table'], chk['key'], chk['expected'], chk['decimals']
    if t == 'summary': got = S[(key[0], key[1])]
    elif t == 'pairs_holm': got = P[(key[0], key[1])]
    else:
        df = pd.read_csv(REPO / t)
        for c, v in key.items(): df = df[df[c].astype(str) == str(v)]
        got = float(df[chk['column']].iloc[0])
    ok = round(float(got), dec) == round(exp, dec) if chk.get('op', 'eq') == 'eq' else float(got) < exp
    report['values'].append(dict(claim=chk['claim'], expected=exp, got=round(float(got), 6), status='PASS' if ok else 'FAIL'))
    if not ok: fail.append(chk['claim'])

print('[5/5] checking plotted figure values ...', flush=True)
r = subprocess.run([sys.executable, 'check_figure_values.py'], cwd=REPO / 'paper_figures' / 'scripts', capture_output=True, text=True)
report['figures'] = 'PASS' if r.returncode == 0 else 'FAIL'
if r.returncode: fail.append('figure values'); print(r.stdout[-1500:], r.stderr[-1500:])

report['overall'] = 'PASS' if not fail else 'FAIL'
(REPO / 'results' / 'reproduction_check.json').write_text(json.dumps(report, indent=2))
for k, v in report['tables'].items(): print(f'  {v[:4]}  table  {k}')
for v in report['values']: print(f"  {v['status']}  value  {v['claim']}: expected {v['expected']}, got {v['got']}")
print(f"  {report['figures']}  figure values")
print('OVERALL:', report['overall'])
sys.exit(0 if not fail else 1)
