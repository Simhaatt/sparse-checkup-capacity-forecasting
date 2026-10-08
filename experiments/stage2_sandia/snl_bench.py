"""Sandia (Preger et al. 2020; batteryarchive.org SNL) second-source capacity benchmark.

Same protocol family as JES round 3 (Codex r3.py + capacity_only/exp_lib.py, copied unmodified):
  * target: capacity at checkups, normalised per cell by its first measured check (SOH = Q/Q0)
  * information for checkup i: rows i-L..i-1 (descriptors + Q); after the origin Q is self-fed
  * grouped 5-fold CV by cell (stratified by chemistry), inner validation = 15% of outer-train
    scoreable cells (seed 1000+fold), selection on equal-cell recursive validation MAPE
  * equal budget: 20 random configurations per family per fold (search schedule), winner refit
    with seeds 42/52/62 (final schedule); trees get the same 20-configuration budget
  * controls: persistence, ARIMA_AICc (prefix only, round-3 univariate.py), lag-one sqrt/linear
    fade fits on equivalent full cycles, all on the same measured prefix
Usage: python snl_bench.py prepare | search | final | controls | analyse      (env SNL_ORIGIN, default 5)
"""
from __future__ import annotations
import itertools, json, os, sys, time, hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ORIGIN = int(os.environ.get('SNL_ORIGIN', '5'))
os.environ['JES_ORIGIN'] = str(ORIGIN)
OUT = HERE / f'origin{ORIGIN}'; OUT.mkdir(exist_ok=True)
FEATS = ['k_exp', 'temperature', 'c_rate_dischg', 'soc_window', 'is_NCA', 'is_NMC', 'Q0']
FAMILIES = ['compact_lstm', 'gru', 'large_lstm', 'xgb', 'rf']
SEEDS = [42, 52, 62]
CHEM = {'LFP': 1, 'NCA': 2, 'NMC': 3}


def data():
    o = pd.read_csv(HERE.parent / 'snl_checkups.csv')
    o = o[o.groupby('cell_id').first_cycle.transform('min').eq(1)].copy()   # Q0 must be the initial check
    o['Q'] = o.SOH
    # Planned throughput only: cycle number x nominal SOC window. NOT measured cumulative Ah, which for
    # 0-100% cells depends on (future) capacity and would leak it into later windows.
    o['k_exp'] = (o.first_cycle - 1) * (o.soc_max - o.soc_min) / 100.0
    o['timestamp_s'] = o.t_days * 86400
    o['c_rate_chg'] = o.crate_c
    o['c_rate_dischg'] = o.crate_d
    o['soc_window'] = o.soc_max - o.soc_min
    o['is_NCA'] = (o.cathode == 'NCA').astype(float)
    o['is_NMC'] = (o.cathode == 'NMC').astype(float)
    o['Q0'] = o.groupby('cell_id').Q0.transform('first')
    o['age_type'] = o.cathode.map(CHEM)
    return o.sort_values(['cell_id', 'checkup_idx']).reset_index(drop=True)


def scoreable(d):
    n = d.groupby('cell_id').size(); return set(n[n > ORIGIN].index)


def prepare():
    d = data(); cells = d.groupby('cell_id').cathode.first()
    rng = np.random.default_rng(20261007); fold = {}
    for chem, idx in cells.groupby(cells):
        ids = np.array(sorted(idx.index)); rng.shuffle(ids)
        for j, c in enumerate(ids):
            fold[c] = j % 5 + 1
    sc = scoreable(d); parts = []
    for f in range(1, 6):
        test = {c for c, v in fold.items() if v == f}; rest = set(cells.index) - test
        elig = np.array(sorted(sc & rest)); r = np.random.default_rng(1000 + f)
        val = set(r.choice(elig, size=int(round(.15 * len(rest))), replace=False))
        for c in sorted(cells.index):
            role = 'test' if c in test else 'validation' if c in val else 'fit'
            parts.append(dict(fold=f, cell_id=c, role=role, scored=c in sc, cathode=cells[c]))
    pd.DataFrame(parts).to_csv(OUT / 'partitions.csv', index=False)
    Ls = [l for l in (2, 3, 4, 5) if l <= ORIGIN]
    spaces = {'compact_lstm': dict(L=Ls, hidden=[16, 32, 48, 64], layers=[1], H=[1, 2, 3, 5]),
              'gru': dict(L=Ls, hidden=[32, 64, 96, 128], layers=[1, 2], H=[1, 2, 3, 5]),
              'large_lstm': dict(L=Ls, hidden=[128, 160, 192, 256], layers=[1, 2], H=[1, 2, 3, 5]),
              'xgb': dict(L=Ls, n_estimators=[100, 200, 300], depth=[2, 3, 4, 6], subsample=[.7, .85, 1.], colsample=[.7, .85, 1.]),
              'rf': dict(L=Ls, n_estimators=[100, 200, 300], depth=[4, 8, 12, None], leaf=[1, 2, 4, 8], max_features=[.5, .75, 1.])}
    trials = []
    for fi, fam in enumerate(FAMILIES):
        r = np.random.default_rng(20261005 + fi * 100); seen = set()
        while len(seen) < 20:
            cfg = {}
            for k, v in spaces[fam].items():
                x = v[r.integers(len(v))]; cfg[k] = x
            if fam in ('compact_lstm', 'gru', 'large_lstm'):
                cfg['lr'] = float(np.exp(r.uniform(np.log(3e-4), np.log(3e-3))))
            elif fam == 'xgb':
                cfg['lr'] = float(np.exp(r.uniform(np.log(.02), np.log(.2))))
            key = json.dumps(cfg, sort_keys=True)
            if key in seen:
                continue
            seen.add(key); trials.append(dict(family=fam, trial=len(seen), cfg=cfg))
    spec = dict(dataset='Sandia SNL 18650 LFP/NCA/NMC (Preger et al. 2020), batteryarchive.org cycle data; checks extracted by extract_checkups.py',
                origin=ORIGIN, cells=int(cells.size), scored_cells=len(sc), features=FEATS, target='SOH=Q/Q0 at capacity checks',
                folds='5 grouped folds, stratified by chemistry, rng 20261007', validation='15% of outer-train scoreable cells, rng 1000+fold',
                budget='20 configs/family/fold, search schedule; winner x seeds 42/52/62 final schedule', spaces=spaces, trials=trials,
                data_sha256=hashlib.sha256((HERE.parent / 'snl_checkups.csv').read_bytes()).hexdigest())
    (OUT / 'PLAN.json').write_text(json.dumps(spec, indent=2, default=str))
    print(f'cells {cells.size}, scored {len(sc)}; per fold test/val/fit:',
          pd.DataFrame(parts).groupby(['fold', 'role']).size().unstack().to_dict('index'))


def split(f, d):
    p = pd.read_csv(OUT / 'partitions.csv'); p = p[p.fold.eq(f)].set_index('cell_id').role
    return {r: d[d.cell_id.map(p).eq(r)].copy() for r in ('fit', 'validation', 'test')}


def equal_cell(p):
    return float((100 * abs(p.Q_pred - p.Q_true) / p.Q_true).groupby(p.cell_id).mean().mean())


def tree_model(fam, cfg, seed):
    if fam == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=cfg['n_estimators'], max_depth=cfg['depth'], min_samples_leaf=cfg['leaf'],
                                     max_features=cfg['max_features'], n_jobs=1, random_state=seed)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=cfg['n_estimators'], max_depth=cfg['depth'], learning_rate=cfg['lr'], subsample=cfg['subsample'],
                        colsample_bytree=cfg['colsample'], n_jobs=1, random_state=seed)


def predict_tree(m, d, L):
    rows = []; cols = FEATS + ['Q']
    for cid, g0 in d.groupby('cell_id'):
        g = g0.sort_values('k_exp', kind='mergesort').reset_index(drop=True); buf = g[cols].to_numpy(float).copy()
        for i in range(ORIGIN, len(g)):
            p = max(1e-6, float(m.predict(buf[i - L:i].ravel()[None])[0]))
            rows.append(dict(cell_id=cid, row=i, step=i - ORIGIN + 1, age_type=int(g.age_type.iloc[i]), Q_true=g.Q.iloc[i], Q_pred=p))
            buf[i, -1] = p
    return pd.DataFrame(rows)


def fit_one(job):
    import torch; torch.set_num_threads(1)
    sys.path.insert(0, str(HERE)); import exp_lib as X
    X.COMMON_START = ORIGIN
    d = data(); s = split(job['fold'], d); fam, cfg, seed = job['family'], job['cfg'], job['seed']
    t = time.perf_counter(); out = dict(job_id=job['job_id'], family=fam, fold=job['fold'], seed=seed, stage=job['stage'], cfg_json=json.dumps(cfg, sort_keys=True))
    if fam in ('compact_lstm', 'gru', 'large_lstm'):
        row, pred, (m, sx, sy) = X.train_eval(s['fit'], s['validation'], s['test'] if job['stage'] == 'final' else None,
                                               'gru' if fam == 'gru' else 'lstm', cfg, FEATS, seed, job['schedule'])
        yt, yp, meta, _ = X.rollout_predict(m, sx, sy, s['validation'], FEATS, cfg['L'])
        v = meta.copy(); v['Q_true'] = yt[:, 0]; v['Q_pred'] = yp[:, 0]
        out['parameters'] = X.count_params(m); out['weights_bytes'] = X.serialized_bytes(m)
        if pred is not None:
            pred = pred.rename(columns={})
    else:
        xx, yy = X.build_sequences(s['fit'], FEATS, cfg['L'])
        m = tree_model(fam, cfg, seed); m.fit(xx.reshape(len(xx), -1), yy.ravel())
        v = predict_tree(m, s['validation'], cfg['L'])
        pred = predict_tree(m, s['test'], cfg['L']) if job['stage'] == 'final' else None
        import pickle; out['weights_bytes'] = len(pickle.dumps(m))
    out['val_equal_cell_MAPE'] = equal_cell(v); out['fit_wall_seconds'] = time.perf_counter() - t
    if pred is not None:
        pred = pred[['cell_id', 'row', 'step', 'age_type', 'Q_true', 'Q_pred']].copy()
        for k in ('job_id', 'family', 'fold', 'seed'):
            pred[k] = job[k]
        out['test_equal_cell_MAPE'] = equal_cell(pred)
    return out, pred


def append(p, df):
    df.to_csv(p, mode='a', header=not p.exists(), index=False)


def run(jobs, runs, preds):
    from concurrent.futures import ProcessPoolExecutor
    done = set(pd.read_csv(runs).job_id) if runs.exists() else set()
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'{len(todo)} jobs to run ({len(done)} done)', flush=True)
    with ProcessPoolExecutor(int(os.environ.get('SNL_WORKERS', '2')), max_tasks_per_child=20) as ex:
        for k, (row, pred) in enumerate(ex.map(fit_one, todo)):
            append(runs, pd.DataFrame([row]))
            if pred is not None:
                append(preds, pred)
            print(k + 1, row['job_id'], round(row['val_equal_cell_MAPE'], 4), round(row['fit_wall_seconds'], 1), flush=True)


def search():
    spec = json.loads((OUT / 'PLAN.json').read_text())
    jobs = [dict(job_id=f"s_{t['family']}_t{t['trial']}_f{f}", family=t['family'], cfg=t['cfg'], fold=f, seed=42, stage='search', schedule='search')
            for f in range(1, 6) for t in spec['trials']]
    run(jobs, OUT / 'search_runs.csv', OUT / 'search_pred.csv')


def final():
    s = pd.read_csv(OUT / 'search_runs.csv')
    s['trial'] = s.job_id.str.extract(r'_t(\d+)_').astype(int)
    win = s.sort_values(['val_equal_cell_MAPE', 'trial']).groupby(['family', 'fold']).head(1)
    win[['family', 'fold', 'trial', 'cfg_json', 'val_equal_cell_MAPE']].to_csv(OUT / 'fold_selected_settings.csv', index=False)
    jobs = [dict(job_id=f"f_{r.family}_f{r.fold}_s{sd}", family=r.family, cfg=json.loads(r.cfg_json), fold=int(r.fold), seed=sd, stage='final', schedule='final')
            for r in win.itertuples() for sd in SEEDS]
    run(jobs, OUT / 'final_runs.csv', OUT / 'final_pred.csv')


def controls():
    sys.path.insert(0, str(HERE)); from univariate import arima_prefix
    d = data(); sc = scoreable(d); frames, logs = [], []
    for cid, g0 in d[d.cell_id.isin(sc)].groupby('cell_id'):
        g = g0.sort_values('k_exp', kind='mergesort').reset_index(drop=True); y = g.Q.to_numpy(); n = len(g); h = n - ORIGIN
        efc = g.k_exp.to_numpy(); prev = np.maximum(np.arange(n) - 1, 0)
        fc = {'persistence': np.full(h, y[ORIGIN - 1])}
        a, log = arima_prefix(y[:ORIGIN], h); fc['ARIMA_AICc'] = a; logs.append(dict(cell_id=cid, **log))
        for name, ax in (('sqrt_efc_lag1', np.sqrt(efc[prev])), ('linear_efc_lag1', efc[prev]), ('sqrt_idx_lag1', np.sqrt(prev.astype(float)))):
            A = np.column_stack([np.ones(ORIGIN), ax[:ORIGIN]]); b = np.linalg.lstsq(A, y[:ORIGIN], rcond=None)[0]; b[1] = min(0., b[1])
            fc[name] = np.maximum(b[0] + b[1] * ax[ORIGIN:], 1e-6)
        for meth, f in fc.items():
            frames.append(pd.DataFrame(dict(cell_id=cid, row=np.arange(ORIGIN, n), step=np.arange(1, h + 1), age_type=g.age_type.iloc[ORIGIN:].to_numpy(),
                                            Q_true=y[ORIGIN:], Q_pred=f, family=meth, seed=0)))
    pd.concat(frames).to_csv(OUT / 'controls_pred.csv', index=False); pd.DataFrame(logs).to_csv(OUT / 'arima_logs.csv', index=False)
    print('controls done', len(sc), 'cells')


def analyse():
    from scipy import stats
    f = pd.read_csv(OUT / 'final_pred.csv'); c = pd.read_csv(OUT / 'controls_pred.csv')
    allp = pd.concat([f[['cell_id', 'step', 'age_type', 'Q_true', 'Q_pred', 'family', 'seed']], c], ignore_index=True)
    allp['ape'] = 100 * abs(allp.Q_pred - allp.Q_true) / allp.Q_true
    cell = allp.groupby(['family', 'seed', 'cell_id']).ape.mean().groupby(['family', 'cell_id']).mean().unstack(0)
    types = allp.groupby('cell_id').age_type.first().map({v: k for k, v in CHEM.items()})
    def boot(x):
        b = np.random.default_rng(20261005).choice(x, (10000, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])
    rows = []
    for m in cell.columns:
        x = cell[m].dropna().to_numpy(); lo, hi = boot(x)
        rows.append(dict(method=m, n_cells=len(x), equal_cell_MAPE=x.mean(), ci95_low=lo, ci95_high=hi,
                         **{f'MAPE_{t}': cell.loc[types == t, m].mean() for t in CHEM}))
    summ = pd.DataFrame(rows).sort_values('equal_cell_MAPE'); summ.to_csv(OUT / 'summary.csv', index=False)
    pairs = []
    for ref in ('ARIMA_AICc', 'persistence', 'sqrt_efc_lag1'):
        fam = []
        for m in cell.columns:
            if m == ref:
                continue
            dd = (cell[m] - cell[ref]).dropna(); lo, hi = boot(dd.to_numpy())
            fam.append(dict(family=f'snl_origin{ORIGIN}_vs_{ref}', comparison=f'{m} - {ref}', n=len(dd), mean_diff=dd.mean(), ci95_low=lo, ci95_high=hi,
                            p=stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
        fam = pd.DataFrame(fam); o = np.argsort(fam.p.to_numpy()); adj = np.empty(len(fam)); run_ = 0
        for r_, i in enumerate(o):
            run_ = max(run_, (len(fam) - r_) * fam.p.iloc[i]); adj[i] = min(1, run_)
        fam['p_holm'] = adj; pairs.append(fam)
    pd.concat(pairs).to_csv(OUT / 'paired_tests.csv', index=False)
    hz = allp.groupby(['family', 'step']).ape.mean().unstack(0); hz.to_csv(OUT / 'error_by_step.csv')
    pd.set_option('display.width', 200)
    print(summ.round(3).to_string(index=False)); print(pd.concat(pairs).round(4).to_string(index=False))


if __name__ == '__main__':
    {'prepare': prepare, 'search': search, 'final': final, 'controls': controls, 'analyse': analyse}[sys.argv[1]]()
