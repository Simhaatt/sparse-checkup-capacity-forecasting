"""Round 4 C: does pooling public datasets help a capacity-history-only (SOH) forecaster?
Arms per test dataset/fold/origin/family/seed: source_only vs pooled (+ all cells of the other public datasets).
Luh-Blank: configuration = R4A validation-best history-only trial of that family (origin, fold); train fit+calibration
(+validation for XGB), neural early stopping on validation. Sandia: configuration = Sandia fold-selected setting of
that family at that origin, history-only input; train fit (+validation for XGB). Usage: python r4_pooled.py [--workers 2]"""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
from r3 import *
import copy, argparse, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, _SNL)
FAMS = ['gru', 'xgb']
COLS = ['cell_id', 'timestamp_s', 'Q', 'k_exp', 'age_type']


def luh():
    d = data().copy(); d['Q'] = d.Q / d.groupby('cell_id').Q.transform('first'); d['cell_id'] = 'LB:' + d.cell_id; return d
def snl():
    import snl_bench as S
    d = S.data().copy(); d['cell_id'] = 'SNL:' + d.cell_id; return d
def oxf():
    d = pd.read_csv(ROOT / 'inputs/oxford_capacity_checkups.csv'); d['Q'] = d.Q_SOH; d['cell_id'] = 'OX:' + d.cell_id; return d
def kir():
    d = pd.read_csv(ROOT / 'inputs/kirkaldy_40_normalized_features.csv'); d['cell_id'] = 'KIR:' + d.cell_key.astype(str); d['timestamp_s'] = d.rpt_idx.astype(float); d['Q'] = d.SOH; d['k_exp'] = d.ageing_cycles.astype(float); return d
LOADERS = dict(LB=luh, SNL=snl, OX=oxf, KIR=kir)


def roles_for(target, fold, origin):
    if target == 'LB':
        m = pd.read_csv(ROOT / 'audit/cell_partitions.csv'); m = m[m.fold.eq(fold)]
        r = dict(zip('LB:' + m.cell_id, m.role)); return {k: ('fit' if v == 'calibration' else v) for k, v in r.items()}
    m = pd.read_csv(f'{_SNL}/origin{origin}/partitions.csv'); m = m[m.fold.eq(fold)]
    return dict(zip('SNL:' + m.cell_id, m.role))


def fit_job(j):
    import torch, exp_lib as X
    torch.set_num_threads(1); X.COMMON_START = j['origin']
    frames = {k: f()[COLS] for k, f in LOADERS.items()}
    roles = roles_for(j['target'], j['fold'], j['origin']); T = frames[j['target']]; r = T.cell_id.map(roles)
    tr, va, te = T[r.eq('fit')], T[r.eq('validation')], T[r.eq('test')]
    n = T.groupby('cell_id').size(); long = set(n[n > j['origin']].index); te = te[te.cell_id.isin(long)]; va = va[va.cell_id.isin(long)]
    if j['family'] == 'xgb': tr = pd.concat([tr, va])
    if j['arm'] == 'pooled': tr = pd.concat([tr] + [frames[k] for k in frames if k != j['target']], ignore_index=True)
    cfg = copy.deepcopy(j['cfg']); cfg.pop('input', None); cfg['L'] = min(cfg['L'], j['origin']); t0 = time.perf_counter()
    if j['family'] == 'gru':
        _, _, (m, sx, sy) = X.train_eval(tr, va, None, 'gru', cfg, [], j['seed'], 'final'); p = predict_neural(X, m, sx, sy, te, [], cfg['L'])
    else:
        xx, yy = X.build_sequences(tr, [], cfg['L']); m = tree_model('xgb', cfg, j['seed']); m.fit(xx.reshape(len(xx), -1), yy.ravel()); p = predict_tree(m, te, [], cfg['L'], j['origin'])
    p = p[['cell_id', 'row', 'step', 'Q_true', 'Q_pred']].copy()
    for k in ('job_id', 'target', 'arm', 'family', 'fold', 'seed', 'origin'): p[k] = j[k]
    return dict(job_id=j['job_id'], train_cells=tr.cell_id.nunique(), wall=time.perf_counter() - t0, cfg_json=json.dumps(cfg, sort_keys=True)), p


def jobs():
    J = []
    s = pd.read_csv(ROOT / 'results/r4_search_runs.csv'); s = s[s.input.eq('history_only')]
    for o in (10, 15):  # R4A_compute_amendment
        for f in range(1, 6):
            for fam in FAMS:
                g = s[(s.origin == o) & (s.fold == f) & (s.family == fam)].sort_values(['val_Q_MAPE_equal_cell', 'trial'], kind='mergesort')
                cfg = json.loads(g.iloc[0].cfg_json)
                for arm in ('source_only', 'pooled'):
                    for sd in SEEDS: J.append(dict(job_id=f'LB_o{o}_f{f}_{fam}_{arm}_s{sd}', target='LB', origin=o, fold=f, family=fam, arm=arm, seed=sd, cfg=cfg))
    for o in (5, 8):
        w = pd.read_csv(f'{_SNL}/origin{o}/fold_selected_settings.csv')
        for f in range(1, 6):
            for fam in FAMS:
                cfg = json.loads(w[(w.fold == f) & (w.family == fam)].iloc[0].cfg_json)
                for arm in ('source_only', 'pooled'):
                    for sd in SEEDS: J.append(dict(job_id=f'SNL_o{o}_f{f}_{fam}_{arm}_s{sd}', target='SNL', origin=o, fold=f, family=fam, arm=arm, seed=sd, cfg=cfg))
    return J


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); a = ap.parse_args()
    rp, pp = ROOT / 'results/r4_pooled_runs.csv', ROOT / 'results/r4_pooled_pred.csv'
    J = jobs(); done = set(pd.read_csv(rp).job_id) if rp.exists() else set(); todo = [j for j in J if j['job_id'] not in done]
    # cheap Sandia first? order: as listed
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=4) as ex:
        for k, (row, p) in enumerate(ex.map(fit_job, todo)):
            append(pp, p); append(rp, pd.DataFrame([row])); print('DONE pooled', len(done) + k + 1, '/', len(J), row['job_id'], row['train_cells'], round(row['wall'], 1), flush=True)
    print('R4_POOLED_COMPLETE', flush=True)
