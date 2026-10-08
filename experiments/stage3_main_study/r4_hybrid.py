"""Round 4 D: sqrt-fade prior + cross-cell XGBoost residual learner (direct multi-step). Luh-Blank origins 10/15/20.
Prior: per-cell lag-one sqrt fit on k (t_days when k is constant in the prefix), slope<=0 (round-3 sqrt_k_lag1 convention).
Usage: python r4_hybrid.py luh   (Sandia variant: r4_hybrid_snl.py)"""
from r3 import *
import itertools
from xgboost import XGBRegressor
CONFIGS = [dict(max_depth=3, n_estimators=200, learning_rate=.05), dict(max_depth=3, n_estimators=400, learning_rate=.05),
           dict(max_depth=5, n_estimators=200, learning_rate=.05), dict(max_depth=5, n_estimators=400, learning_rate=.03),
           dict(max_depth=4, n_estimators=300, learning_rate=.1), dict(max_depth=6, n_estimators=300, learning_rate=.05)]
STATIC = ['temperature', 'c_rate_chg', 'c_rate_dischg', 'soc_window', 'age_type']


def prior(y, axis, o):
    """lag-one sqrt fit on prefix y[:o]; returns fitted values for all rows (index-aligned)."""
    n = len(y); prev = np.maximum(np.arange(n) - 1, 0); x = np.sqrt(np.maximum(axis[prev], 0))
    A = np.column_stack([np.ones(o), x[:o]]); b = np.linalg.lstsq(A, y[:o], rcond=None)[0]; b[1] = min(0., b[1])
    return np.maximum(b[0] + b[1] * x, 1e-6), x


def cell_rows(g, origins, static_cols, kcol='k_exp'):
    g = ordered(g); soh = (g.Q / g.Q.iloc[0]).to_numpy(); n = len(g)
    k = g[kcol].to_numpy(float); days = ((g.timestamp_s - g.timestamp_s.iloc[0]) / 86400).to_numpy()
    rows = []
    for o in origins:
        if o >= n: continue
        axis = k if np.ptp(k[:o]) > 0 else days
        pr, x = prior(soh, axis, o); res_pref = soh[:o] - pr[:o]
        slope = (soh[o - 1] - soh[max(0, o - 4)]) / max(1, min(3, o - 1))
        for i in range(o, n):
            rows.append(dict(cell_id=g.cell_id.iloc[0], row=i, step=i - o + 1, origin=o, prior=pr[i], last_soh=soh[o - 1], last_res=res_pref[-1],
                             mean_abs_res=np.abs(res_pref).mean(), slope=slope, gap=x[i] - x[o - 1], k_prev=k[i - 1], soh_true=soh[i], Q0=g.Q.iloc[0],
                             age_type=int(g.age_type.iloc[i]), **{c: float(g[c].iloc[0]) for c in static_cols}))
    return rows


FEAT = ['step', 'origin', 'prior', 'last_soh', 'last_res', 'mean_abs_res', 'slope', 'gap', 'k_prev', 'Q0'] + STATIC


def frame(d, origins):
    return pd.DataFrame([r for _, g in d.groupby('cell_id') for r in cell_rows(g, origins, [c for c in STATIC if c != 'age_type'])])


def ecm(p):
    return float((100 * (p.pred - p.soh_true).abs() / p.soh_true).groupby(p.cell_id).mean().mean())


def main():
    d = data(); long = cohort(d); out = []; sel = []
    for fold in range(1, 6):
        s = split(fold, d); tr = pd.concat([s['fit'], s['calibration']]); maxn = tr.groupby('cell_id').size().max()
        T = frame(tr, list(range(8, maxn - 1, 2)))
        for o in (10, 15, 20):
            V = frame(s['validation'][s['validation'].cell_id.isin(long)], [o]); E = frame(s['test'][s['test'].cell_id.isin(long)], [o])
            scores = []
            for ci, c in enumerate(CONFIGS):
                m = XGBRegressor(**c, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=42).fit(T[FEAT], T.soh_true - T.prior)
                V['pred'] = np.maximum(V.prior + m.predict(V[FEAT]), 1e-6); scores.append(ecm(V))
            best = int(np.argmin(scores)); sel.append(dict(fold=fold, origin=o, config=best, **CONFIGS[best], val_MAPE=scores[best], train_rows=len(T)))
            for sd in SEEDS:
                m = XGBRegressor(**CONFIGS[best], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[FEAT], T.soh_true - T.prior)
                for role, F in (('validation', V), ('test', E)):
                    f = F[['cell_id', 'row', 'step', 'age_type', 'soh_true', 'prior', 'Q0']].copy(); f['pred'] = np.maximum(F.prior + m.predict(F[FEAT]), 1e-6)
                    f['Q_true'] = f.soh_true * f.Q0; f['Q_pred'] = f.pred * f.Q0; f['origin'] = o; f['fold'] = fold; f['seed'] = sd; f['role'] = role; f['method'] = 'hybrid_sqrt_xgb'
                    out.append(f)
            print('fold', fold, 'origin', o, 'val', round(scores[best], 3), flush=True)
    pd.concat(out).to_csv(ROOT / 'results/r4_hybrid_pred.csv', index=False); pd.DataFrame(sel).to_csv(ROOT / 'tables/r4_hybrid_selected.csv', index=False)
    p = pd.concat(out); t = p[p.role.eq('test')]
    for o in (10, 15, 20):
        x = t[t.origin.eq(o)]; ape = 100 * (x.Q_pred - x.Q_true).abs() / x.Q_true
        print('origin', o, 'hybrid equal-cell', round(ape.groupby([x.seed, x.cell_id]).mean().groupby(level=1).mean().mean(), 3),
              'prior-only', round((100 * (x.prior - x.soh_true).abs() / x.soh_true).groupby(x.cell_id).mean().mean(), 3))


if __name__ == '__main__':
    main()
