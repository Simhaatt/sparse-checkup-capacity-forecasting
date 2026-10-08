"""Post-hoc summaries for R4C (pooled) and R4F (fine-tuned Chronos). Equal-cell MAPE: steps -> seeds -> cells.
Paired Wilcoxon on common cells; Holm within (dataset, origin) family. Usage: python r4_post.py pooled|chronos"""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parent; T = ROOT / 'tables'; R = ROOT / 'results'; SNL = Path(_SNL)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o): r = max(r, (len(p) - i) * p[j]); a[j] = min(1, r)
    return a


def cellmape(df, keys):
    """df has method, origin, seed, cell_id, true, pred -> Series indexed (method, origin, cell_id)."""
    ape = 100 * (df.pred - df.true).abs() / df.true
    return ape.groupby([df[k] for k in keys + ['seed', 'cell_id']]).mean().groupby(level=keys + ['cell_id']).mean()


def refs(dataset):
    """Hybrid and ARIMA test predictions in SOH units, columns method, origin, seed, cell_id, row, true, pred."""
    if dataset == 'luh':
        h = pd.read_csv(R / 'r4_hybrid_pred.csv'); h = h[h.role.eq('test')]
        h = h.assign(method='hybrid_sqrt_xgb', true=h.soh_true, pred=h.pred)
        a = pd.read_csv(ROOT / 'inputs/r3_reference_predictions.csv'); a = a[a.role.eq('test') & a.method.eq('ARIMA_AICc')]
        a = a.assign(true=a.Q_true / a.Q0, pred=a.Q_pred / a.Q0)
    else:
        h = pd.read_csv(R / 'r4_hybrid_snl_pred.csv'); h = h[h.role.eq('test')]
        h = h.assign(method='hybrid_sqrt_xgb', true=h.soh_true, pred=h.pred)
        a = []
        for o in (5, 8):
            c = pd.read_csv(SNL / f'origin{o}/controls_pred.csv'); c = c[c.family.eq('ARIMA_AICc')].assign(origin=o); a.append(c)
        a = pd.concat(a); a = a.assign(method='ARIMA_AICc', true=a.Q_true, pred=a.Q_pred)
    cols = ['method', 'origin', 'seed', 'cell_id', 'row', 'true', 'pred']
    return pd.concat([h[cols], a[cols]], ignore_index=True)


def compare(cell, target_methods, ref_methods, dataset):
    rows = []
    for o in sorted(cell.index.get_level_values('origin').unique()):
        c = cell.xs(o, level='origin').unstack(0); fam = []
        for m in target_methods:
            for r in ref_methods:
                if m not in c or r not in c or m == r: continue
                d = (c[m] - c[r]).dropna()
                if len(d) < 5: continue
                fam.append(dict(dataset=dataset, origin=o, comparison=f'{m} - {r}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        if fam:
            fam = pd.DataFrame(fam); fam['p_holm'] = holm(fam.p); rows.append(fam)
    return pd.concat(rows) if rows else pd.DataFrame()


def pooled():
    p = pd.read_csv(R / 'r4_pooled_pred.csv'); p['method'] = p.family + '_' + p.arm
    S, P = [], []
    for tgt, ds in (('LB', 'luh'), ('SNL', 'snl')):
        x = p[p.target.eq(tgt)].copy(); x['cell_id'] = x.cell_id.str.split(':', n=1).str[1]; x = x.assign(true=x.Q_true, pred=x.Q_pred)
        if x.empty: continue
        ref = refs(ds); ref = ref[ref.origin.isin(x.origin.unique()) & ref.cell_id.isin(x.cell_id.unique())]
        x = x[x.cell_id.isin(ref.loc[ref.method.eq('ARIMA_AICc'), 'cell_id'])]  # same scored cohort as the main benchmark
        allp = pd.concat([x[['method', 'origin', 'seed', 'cell_id', 'row', 'true', 'pred']], ref], ignore_index=True)
        cell = cellmape(allp, ['method', 'origin'])
        s = cell.groupby(level=['method', 'origin']).agg(['mean', 'size']).reset_index().rename(columns={'mean': 'MAPE', 'size': 'n_cells'}); s['dataset'] = ds; S.append(s)
        rows = []
        for o in sorted(x.origin.unique()):
            c = cell.xs(o, level='origin').unstack(0); fam = []
            for f in ('gru', 'xgb'):
                for a, b in ((f'{f}_pooled', f'{f}_source_only'), (f'{f}_pooled', 'hybrid_sqrt_xgb'), (f'{f}_source_only', 'hybrid_sqrt_xgb')):
                    if a in c and b in c:
                        d = (c[a] - c[b]).dropna(); fam.append(dict(dataset=ds, origin=o, comparison=f'{a} - {b}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
            fam = pd.DataFrame(fam); fam['p_holm'] = holm(fam.p); rows.append(fam)
        P.append(pd.concat(rows))
    S = pd.concat(S); P = pd.concat(P)
    S.pivot_table(index=['dataset', 'method'], columns='origin', values='MAPE').reset_index().to_csv(T / 'r4_pooled_summary.csv', index=False)
    P.to_csv(T / 'r4_pooled_pairs.csv', index=False)
    print(S.pivot_table(index=['dataset', 'method'], columns='origin', values='MAPE').round(3).to_string()); print(P.round(4).to_string(index=False))


def chronos():
    S, P = [], []
    for ds in ('snl', 'luh'):
        fs = sorted(R.glob(f'r4_chronos_ft_{ds}*_pred.csv'))
        if not fs: continue
        c = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); c['seed'] = 0; c = c.assign(true=c.soh_true)
        cov = ((c.soh_true >= c.q10) & (c.soh_true <= c.q90)).groupby([c.method, c.origin, c.cell_id]).mean().groupby(level=[0, 1]).mean()
        ref = refs(ds); ref = ref[ref.cell_id.isin(c.cell_id.unique()) & ref.origin.isin(c.origin.unique())]
        # alignment check: same truth on common rows
        chk = c.merge(ref[ref.method.eq('hybrid_sqrt_xgb') & ref.seed.eq(42)], on=['origin', 'cell_id', 'row'], suffixes=('', '_h'))
        assert len(chk) and np.allclose(chk.true, chk.true_h, atol=1e-6), 'truth misaligned'
        if ds == 'snl':
            z = []
            for o in (5, 8):
                q = pd.read_csv(SNL / f'origin{o}/chronos_pred.csv').assign(origin=o); z.append(q.assign(method=q.family, true=q.Q_true, pred=q.Q_pred))
            z = pd.concat(z); ref = pd.concat([ref, z[z.cell_id.isin(c.cell_id.unique())][ref.columns]])
        allp = pd.concat([c[['method', 'origin', 'seed', 'cell_id', 'row', 'true', 'pred']], ref], ignore_index=True)
        cell = cellmape(allp, ['method', 'origin'])
        s = cell.groupby(level=['method', 'origin']).mean().rename('MAPE').reset_index(); s['dataset'] = ds
        s['q10_q90_coverage'] = [cov.get((m, o), np.nan) for m, o in zip(s.method, s.origin)]; S.append(s)
        P.append(compare(cell, sorted(c.method.unique()), ['hybrid_sqrt_xgb', 'ARIMA_AICc'] + [m for m in s.method.unique() if m.endswith('_zero')], ds))
    S = pd.concat(S); P = pd.concat(P)
    S.to_csv(T / 'r4_chronos_ft_summary.csv', index=False); P.to_csv(T / 'r4_chronos_ft_pairs.csv', index=False)
    print(S.round(3).to_string(index=False)); print(P.round(4).to_string(index=False))


if __name__ == '__main__':
    {'pooled': pooled, 'chronos': chronos}[sys.argv[1]]()
