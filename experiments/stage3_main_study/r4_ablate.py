"""R4F: hybrid ablations V1-V6 (incl. Xuan-et-al.-style ARIMA prior) on Luh-Blank (origins 10/15/20) and Sandia (5/8).
Usage: python r4_ablate.py arima_cache | luh | snl"""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, json, pickle, time
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, _SNL)
import r3
from univariate import arima_prefix
from xgboost import XGBRegressor
CONFIGS = [dict(max_depth=3, n_estimators=200, learning_rate=.05), dict(max_depth=3, n_estimators=400, learning_rate=.05),
           dict(max_depth=5, n_estimators=200, learning_rate=.05), dict(max_depth=5, n_estimators=400, learning_rate=.03),
           dict(max_depth=4, n_estimators=300, learning_rate=.1), dict(max_depth=6, n_estimators=300, learning_rate=.05)]
SEEDS = [42, 52, 62]
DIAG = ['last_res', 'mean_abs_res']
VARIANTS = {'V1_main': ('sqrt', True, 'all'), 'V2_no_diag': ('sqrt', False, 'all'), 'V3_single_origin': ('sqrt', True, 'single'),
            'V4_linear_prior': ('linear', True, 'all'), 'V5_persistence_anchor': ('persist', False, 'all'), 'V6_xuan_style': ('arima', False, 'every4')}


def ordered(g): return g.sort_values('timestamp_s', kind='mergesort').reset_index(drop=True)


def dataset(name):
    if name == 'luh':
        d = r3.data().copy(); static = ['temperature', 'c_rate_chg', 'c_rate_dischg', 'soc_window', 'age_type']
    else:
        import snl_bench as S
        d = S.data().copy(); d['Q'] = d.Q * d.Q0; static = ['temperature', 'c_rate_chg', 'c_rate_dischg', 'soc_window', 'age_type']
    return d, static


def arima_cache(name, origins_of):
    path = ROOT / f'results/arima_cache_{name}.pkl'
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    d, _ = dataset(name); t0 = time.time(); k = 0
    for cid, g in d.groupby('cell_id'):
        g = ordered(g); y = (g.Q / g.Q.iloc[0]).to_numpy(); n = len(y)
        for o in origins_of(n):
            if (cid, o) in cache or o >= n: continue
            cache[(cid, o)] = arima_prefix(y[:o], n - o)[0]; k += 1
            if k % 100 == 0:
                path.write_bytes(pickle.dumps(cache)); print('arima', name, k, round(time.time() - t0), flush=True)
    path.write_bytes(pickle.dumps(cache)); print('arima cache', name, len(cache), flush=True); return cache


def cell_rows(g, origins, static, prior_kind, cache):
    g = ordered(g); cid = g.cell_id.iloc[0]; soh = (g.Q / g.Q.iloc[0]).to_numpy(); n = len(soh)
    k = g.k_exp.to_numpy(float); days = ((g.timestamp_s - g.timestamp_s.iloc[0]) / 86400).to_numpy(); prev = np.maximum(np.arange(n) - 1, 0); out = []
    for o in origins:
        if o >= n: continue
        axis = k if np.ptp(k[:o]) > 0 else days; xs = np.sqrt(np.maximum(axis[prev], 0)); xl = axis[prev]
        if prior_kind in ('sqrt', 'linear'):
            x = xs if prior_kind == 'sqrt' else xl
            b = np.linalg.lstsq(np.column_stack([np.ones(o), x[:o]]), soh[:o], rcond=None)[0]; b[1] = min(0., b[1]); pr = np.maximum(b[0] + b[1] * x, 1e-6)
            r = soh[:o] - pr[:o]; lr, mr = r[-1], np.abs(r).mean(); fut = pr[o:]
        elif prior_kind == 'persist':
            fut = np.full(n - o, soh[o - 1]); lr = mr = 0.
        else:
            if (cid, o) not in cache: cache[(cid, o)] = arima_prefix(soh[:o], n - o)[0]  # one-step origin n-1 not pre-cached
            fut = np.maximum(cache[(cid, o)], 1e-6); lr = mr = np.nan
        slope = (soh[o - 1] - soh[max(0, o - 4)]) / max(1, min(3, o - 1))
        for j, i in enumerate(range(o, n)):
            out.append(dict(cell_id=cid, row=i, step=i - o + 1, origin=o, prior=fut[j], last_soh=soh[o - 1], last_res=lr, mean_abs_res=mr, slope=slope,
                            gap=xs[i] - xs[o - 1], k_prev=k[i - 1], soh_true=soh[i], Q0=g.Q.iloc[0], **{c: float(g[c].iloc[i]) for c in static}))
    return out


def feats(static, diag, prior_kind):
    f = ['step', 'origin', 'last_soh', 'slope', 'gap', 'k_prev', 'Q0'] + static
    if prior_kind != 'persist': f = ['prior'] + f
    if diag: f += DIAG
    return f


def frame(d, origins, static, prior_kind, cache):
    return pd.DataFrame([r for _, g in d.groupby('cell_id') for r in cell_rows(g, origins, static, prior_kind, cache)])


def ecm(soh, pred, cells): return float((100 * np.abs(pred - soh) / soh).groupby(cells).mean().mean())


def run(name):
    d, static = dataset(name)
    cache = pickle.loads((ROOT / f'results/arima_cache_{name}.pkl').read_bytes())
    if name == 'luh':
        long = r3.cohort(d); origins = (10, 15, 20)
        def roles(fold):
            s = r3.split(fold, d); return pd.concat([s['fit'], s['calibration']]), s['validation'][s['validation'].cell_id.isin(long)], s['test'][s['test'].cell_id.isin(long)]
        start, every = 8, {'all': 2, 'every4': 4}
    else:
        origins = (5, 8)
        def roles(fold, o=None):
            p = pd.read_csv(f'{_SNL}/origin{o}/partitions.csv'); p = p[p.fold.eq(fold)].set_index('cell_id').role; r = d.cell_id.map(p)
            n = d.groupby('cell_id').size(); sc = set(n[n > o].index)
            return d[r.eq('fit')], d[r.eq('validation') & d.cell_id.isin(sc)], d[r.eq('test') & d.cell_id.isin(sc)]
        start, every = 3, {'all': 1, 'every4': 1}
    out, sel = [], []
    for vname, (pk, diag, mode) in VARIANTS.items():
        F = feats(static, diag, pk)
        for fold in range(1, 6):
            for o in origins:
                tr, va, te = roles(fold) if name == 'luh' else roles(fold, o)
                maxn = tr.groupby('cell_id').size().max()
                torig = [o] if mode == 'single' else list(range(start, maxn - 1, every['every4' if mode == 'every4' else 'all']))
                T = frame(tr, torig, static, pk, cache); V = frame(va, [o], static, pk, cache); E = frame(te, [o], static, pk, cache)
                y = T.soh_true - T.prior if pk != 'persist' else T.soh_true - T.last_soh
                base = (lambda X: X.prior) if pk != 'persist' else (lambda X: X.last_soh)
                sc = []
                for c in CONFIGS:
                    m = XGBRegressor(**c, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=42).fit(T[F], y)
                    sc.append(ecm(V.soh_true, np.maximum(base(V) + m.predict(V[F]), 1e-6), V.cell_id))
                b = int(np.argmin(sc)); sel.append(dict(dataset=name, variant=vname, fold=fold, origin=o, config=b, val_MAPE=sc[b], train_rows=len(T)))
                for sd in SEEDS:
                    m = XGBRegressor(**CONFIGS[b], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[F], y)
                    f = E[['cell_id', 'row', 'step', 'soh_true', 'age_type', 'Q0']].copy(); f['pred'] = np.maximum(base(E) + m.predict(E[F]), 1e-6)
                    f['variant'] = vname; f['fold'] = fold; f['origin'] = o; f['seed'] = sd; out.append(f)
            print(name, vname, 'fold', fold, 'done', flush=True)
    P = pd.concat(out, ignore_index=True); P.to_csv(ROOT / f'results/r4_ablation_{name}_pred.csv', index=False)
    pd.DataFrame(sel).to_csv(ROOT / f'tables/r4_ablation_{name}_selected.csv', index=False)
    P['ape'] = 100 * (P.pred - P.soh_true).abs() / P.soh_true
    s = P.groupby(['variant', 'origin', 'seed', 'cell_id']).ape.mean().groupby(['variant', 'origin', 'cell_id']).mean().groupby(['variant', 'origin']).mean().unstack()
    print(s.round(3).to_string()); s.to_csv(ROOT / f'tables/r4_ablation_{name}_summary.csv')


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'arima_cache':
        arima_cache('luh', lambda n: sorted(set(range(8, n - 1, 4)) | {10, 15, 20}))
        arima_cache('snl', lambda n: sorted(set(range(3, n - 1)) | {5, 8}))
    else:
        run(what)
