"""R4F transfer ablation on Kirkaldy: source sets and persistence-anchor variant (same fixed XGB setting as R4E)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'trainer'))
import r4_pooled as P, r4_transfer as TR
from xgboost import XGBRegressor
from scipy import stats
SOURCES = {'LB': ['luh'], 'SNL': ['snl'], 'LB+SNL': ['luh', 'snl'], 'LB+SNL+OX': ['luh', 'snl', 'oxf']}


def main():
    kir = P.kir()[P.COLS]; base = pd.read_csv(ROOT / 'results/r4_transfer_kirkaldy_pred.csv'); res = []
    fr = {k: getattr(P, k)()[P.COLS] for k in ('luh', 'snl', 'oxf')}
    for sname, parts in SOURCES.items():
        src = pd.concat([fr[p] for p in parts], ignore_index=True)
        T = pd.DataFrame([r for _, g in src.groupby('cell_id') for r in TR.rows(g, range(3, 80))])
        for o in (3, 5, 8):
            E = pd.DataFrame([r for _, g in kir.groupby('cell_id') for r in TR.rows(g, [o])])
            for sd in (42, 52, 62):
                m = XGBRegressor(**TR.CFG, random_state=sd).fit(T[TR.FEAT], T.soh_true - T.prior)
                res.append(E.assign(method=f'hybrid_src_{sname}', seed=sd, pred=np.maximum(E.prior + m.predict(E[TR.FEAT]), 1e-6)))
                if sname == 'LB+SNL+OX':
                    FA = ['step', 'origin', 'last_soh', 'slope', 'gap']
                    m2 = XGBRegressor(**TR.CFG, random_state=sd).fit(T[FA], T.soh_true - T.last_obs)
                    res.append(E.assign(method='anchor_persistence_src_all', seed=sd, pred=np.maximum(E.last_obs + m2.predict(E[FA]), 1e-6)))
        print('source', sname, 'done', flush=True)
    R = pd.concat(res + [base[base.method.isin(['sqrt_idx_prior', 'ARIMA_AICc', 'persistence'])]], ignore_index=True)
    R['ape'] = 100 * (R.pred - R.soh_true).abs() / R.soh_true; R.to_csv(ROOT / 'results/r4_transfer_ablation_pred.csv', index=False)
    cell = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean()
    S = cell.groupby(['method', 'origin']).mean().unstack(); S.to_csv(ROOT / 'tables/r4_transfer_ablation_summary.csv'); print(S.round(3).to_string())
    pairs = []
    for o in (3, 5, 8):
        c = cell.xs(o, level='origin').unstack(0); fam = []
        for m in [x for x in c.columns if x.startswith(('hybrid_src', 'anchor'))]:
            d = (c[m] - c['sqrt_idx_prior']).dropna(); fam.append(dict(origin=o, comparison=f'{m} - sqrt_idx_prior', diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        fam = pd.DataFrame(fam); oo = np.argsort(fam.p.to_numpy()); adj = np.empty(len(fam)); run = 0
        for i, j in enumerate(oo): run = max(run, (len(fam) - i) * fam.p.iloc[j]); adj[j] = min(1, run)
        fam['p_holm'] = adj; pairs.append(fam)
    Pp = pd.concat(pairs); Pp.to_csv(ROOT / 'tables/r4_transfer_ablation_pairs.csv', index=False); print(Pp.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
