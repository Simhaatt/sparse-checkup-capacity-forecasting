"""R4E: unit-free sqrt-prior + residual XGBoost trained on Luh-Blank + Sandia + Oxford, applied unchanged to Kirkaldy."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'trainer'))
import r4_pooled as P
from univariate import arima_prefix
from xgboost import XGBRegressor
from scipy import stats
FEAT = ['step', 'origin', 'prior', 'last_soh', 'last_res', 'mean_abs_res', 'slope', 'gap']
CFG = dict(max_depth=3, n_estimators=200, learning_rate=.05, subsample=.8, colsample_bytree=.8, n_jobs=1)


def rows(g, origins):
    g = g.sort_values('timestamp_s', kind='mergesort').reset_index(drop=True); y = g.Q.to_numpy() / g.Q.iloc[0]; n = len(y); out = []
    prev = np.maximum(np.arange(n) - 1, 0); x = np.sqrt(prev.astype(float))
    for o in origins:
        if o >= n or o < 3: continue
        A = np.column_stack([np.ones(o), x[:o]]); b = np.linalg.lstsq(A, y[:o], rcond=None)[0]; b[1] = min(0., b[1]); pr = np.maximum(b[0] + b[1] * x, 1e-6)
        r = y[:o] - pr[:o]; slope = (y[o - 1] - y[max(0, o - 4)]) / max(1, min(3, o - 1))
        for i in range(o, n):
            out.append(dict(cell_id=g.cell_id.iloc[0], row=i, step=i - o + 1, origin=o, prior=pr[i], last_soh=y[o - 1], last_res=r[-1], mean_abs_res=np.abs(r).mean(),
                            slope=slope, gap=x[i] - x[o - 1], soh_true=y[i], last_obs=y[o - 1]))
    return out


def main():
    src = pd.concat([P.luh(), P.snl(), P.oxf()], ignore_index=True)[P.COLS]
    T = pd.DataFrame([r for _, g in src.groupby('cell_id') for r in rows(g, range(3, 80))])
    kir = P.kir()[P.COLS]; res = []
    for o in (3, 5, 8):
        E = pd.DataFrame([r for _, g in kir.groupby('cell_id') for r in rows(g, [o])])
        for sd in (42, 52, 62):
            m = XGBRegressor(**CFG, random_state=sd).fit(T[FEAT], T.soh_true - T.prior)
            res.append(E.assign(method='hybrid_transfer', seed=sd, pred=np.maximum(E.prior + m.predict(E[FEAT]), 1e-6)))
        res.append(E.assign(method='sqrt_idx_prior', seed=0, pred=E.prior)); res.append(E.assign(method='persistence', seed=0, pred=E.last_obs))
        ar = []
        for cid, g in kir.groupby('cell_id'):
            g = g.sort_values('timestamp_s'); y = (g.Q / g.Q.iloc[0]).to_numpy()
            if len(y) <= o: continue
            f, _ = arima_prefix(y[:o], len(y) - o); ar.append(pd.DataFrame(dict(cell_id=cid, row=np.arange(o, len(y)), pred=f)))
        A = E[['cell_id', 'row', 'step', 'origin', 'soh_true']].merge(pd.concat(ar), on=['cell_id', 'row']); res.append(A.assign(method='ARIMA_AICc', seed=0))
    R = pd.concat(res, ignore_index=True); R['ape'] = 100 * (R.pred - R.soh_true).abs() / R.soh_true
    R.to_csv(ROOT / 'results/r4_transfer_kirkaldy_pred.csv', index=False)
    cell = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean()
    out, pairs = [], []
    for o in (3, 5, 8):
        c = cell.xs(o, level='origin').unstack(0)
        for mth in c.columns: out.append(dict(origin=o, method=mth, n_cells=c[mth].notna().sum(), MAPE=c[mth].mean()))
        fam = []
        for ref in ('sqrt_idx_prior', 'ARIMA_AICc', 'persistence'):
            d = (c['hybrid_transfer'] - c[ref]).dropna(); fam.append(dict(origin=o, comparison=f'hybrid_transfer - {ref}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        fam = pd.DataFrame(fam); o_ = np.argsort(fam.p.to_numpy()); adj = np.empty(len(fam)); run = 0
        for i, j in enumerate(o_): run = max(run, (len(fam) - i) * fam.p.iloc[j]); adj[j] = min(1, run)
        fam['p_holm'] = adj; pairs.append(fam)
    S = pd.DataFrame(out); P_ = pd.concat(pairs); S.to_csv(ROOT / 'tables/r4_transfer_summary.csv', index=False); P_.to_csv(ROOT / 'tables/r4_transfer_pairs.csv', index=False)
    print(S.pivot(index='method', columns='origin', values='MAPE').round(3).to_string()); print(S.groupby('origin').n_cells.max().to_dict()); print(P_.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
