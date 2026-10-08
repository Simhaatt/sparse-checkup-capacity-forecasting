"""Create the run-time folders that the original experiment scripts expect (Level 2).

The main study (experiments/stage3_main_study) reads ROOT/inputs, ROOT/results, ROOT/tables and ROOT/audit.
This script copies the deposited inputs there and decompresses the deposited predictions into results/, so that any
analysis step can be re-run on its own, or the fitting steps can be re-run from scratch.
Usage:  python scripts/setup_workspace.py [--no-results]
"""
import argparse, gzip, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W = REPO / 'experiments' / 'stage3_main_study'
D = REPO / 'data' / 'processed'; P = REPO / 'results' / 'predictions'; T = REPO / 'results' / 'tables' / 'main_study'


def ungz(src, dst):
    with gzip.open(src, 'rb') as f, open(dst, 'wb') as g: shutil.copyfileobj(f, g)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--no-results', action='store_true', help='only inputs (re-run everything from scratch)'); a = ap.parse_args()
    for d in ('inputs', 'results', 'tables', 'audit'): (W / d).mkdir(exist_ok=True)
    for f in ('phase1_cv_all_rows.csv', 'grouped_cv_folds.csv'): shutil.copy(D / 'luh_blank' / f, W / 'inputs' / f)
    shutil.copy(D / 'luh_blank' / 'cell_partitions.csv', W / 'audit' / 'cell_partitions.csv')
    shutil.copy(D / 'kirkaldy' / 'kirkaldy_40_normalized_features.csv', W / 'inputs')
    shutil.copy(D / 'oxford' / 'oxford_capacity_checkups.csv', W / 'inputs')
    for f in ('oxford_cell_errors.csv', 'oxford_chronos_pred.csv'): shutil.copy(P / 'oxford' / f, W / 'inputs' / f)
    ungz(P / 'r3_reference_predictions.csv.gz', W / 'inputs' / 'r3_reference_predictions.csv')   # stage-1 output used as input
    for f in ('robustness_prefixes', 'robustness_cell_errors'): ungz(P / f'{f}.csv.gz', W / 'inputs' / f'{f}.csv')
    if not a.no_results:
        for f in P.glob('r4_*.csv.gz'): ungz(f, W / 'results' / f.name[:-3])
        ungz(P / 'oxford' / 'r4_g1_oxford_pred.csv.gz', W / 'results' / 'r4_g1_oxford_pred.csv')
        for f in P.glob('r4_*_runs.csv'): shutil.copy(f, W / 'results' / f.name)
        for f in T.glob('*.csv'): shutil.copy(f, W / 'tables' / f.name)
    snl = REPO / 'experiments' / 'snl_checkups.csv'
    print('main-study workspace ready:', W)
    print('Sandia checkups', 'found' if snl.exists() else 'NOT found (build it with preprocessing/sandia/extract_checkups.py, see data/README.md)', '->', snl)
