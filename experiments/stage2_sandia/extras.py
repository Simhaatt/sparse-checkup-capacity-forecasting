"""Sandia extras on saved predictions: pre-declared forecast combinations, validation-chosen per-chemistry selector,
cross-fold (CV+-style) split-conformal intervals, decision metrics, and paired tests. Env SNL_ORIGIN (5 or 8).

Rules fixed before looking at test results:
  * best_NN / best_learned are chosen per outer fold by mean inner-validation equal-cell MAPE over seeds.
  * Combinations (equal-weight, computed per seed): C_arima_xgb_nn = mean(ARIMA, XGB, best_NN) [round-3 protocol];
    pairwise means of {ARIMA, XGB, best_NN}; C_sqrt_xgb_nn = mean(sqrt_efc_lag1, XGB, best_NN);
    C_sqrt_bestlearned = mean(sqrt_efc_lag1, best_learned); C_all_learned = mean of the five learned families.
  * Selector: for each outer fold and chemistry, the candidate (all singles + combinations) with the lowest
    inner-validation equal-cell MAPE on that fold's validation cells of that chemistry (>=2 cells, else the fold's
    overall validation winner); applied to that fold's test cells.
  * Conformal: absolute SOH residuals of the same method from the OTHER four folds' test predictions (all seeds),
    pooled within step buckets {1,2,3,4-6,7-12,13+}; rank ceil((n+1)(1-a)); infinite width if rank > n.
    Models for other folds were trained on cells that include this fold's cells, so this is CV+-style, not exact
    split-conformal; stated as such.
  * Decisions: last-step APE; signed last-step error (+ = optimistic, predicted SOH above truth); 80%-SOH crossing
    (first check with SOH <= 0.80 inside the scored horizon) detection F1 over cells and |crossing-step error| for
    true positives.
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import snl_bench as S
OUT = S.OUT; SEEDS = S.SEEDS
LEARNED = ['compact_lstm', 'gru', 'large_lstm', 'xgb', 'rf']; NN = LEARNED[:3]
CHEMNAME = {1: 'LFP', 2: 'NCA', 3: 'NMC'}
KEY = ['cell_id', 'row', 'seed']


def load():
    f = pd.read_csv(OUT / 'final_pred_with_validation.csv')
    c = pd.read_csv(OUT / 'controls_pred.csv')
    ch = OUT / 'chronos_pred.csv'
    if ch.exists():
        c = pd.concat([c, pd.read_csv(ch)[c.columns]], ignore_index=True)
    c = pd.concat([c.assign(seed=s) for s in SEEDS], ignore_index=True)
    part = pd.read_csv(OUT / 'partitions.csv')
    test_fold = part[part.role.eq('test')].set_index('cell_id').fold
    rows = []
    for fold in range(1, 6):
        roles = part[part.fold.eq(fold)].set_index('cell_id').role
        sub = c[c.cell_id.map(roles).isin(['test', 'validation'])].copy()
        sub['fold'] = fold; sub['role'] = sub.cell_id.map(roles); rows.append(sub)
    c = pd.concat(rows, ignore_index=True)
    allp = pd.concat([f, c[f.columns]], ignore_index=True)
    allp['ape'] = 100 * (allp.Q_pred - allp.Q_true).abs() / allp.Q_true
    return allp, test_fold


def ecm(df):  # equal-cell MAPE: steps -> seeds -> cells
    return df.groupby(['seed', 'cell_id']).ape.mean().groupby('cell_id').mean().mean()


def add_derived(allp):
    out = [allp]
    for fold in range(1, 6):
        F = allp[allp.fold.eq(fold)]
        val = F[F.role.eq('validation')]
        vscore = {m: ecm(val[val.family.eq(m)]) for m in LEARNED}
        best_nn = min(NN, key=vscore.get); best_l = min(LEARNED, key=vscore.get)
        piv = F.pivot_table(index=KEY + ['step', 'role'], columns='family', values='Q_pred').reset_index()
        truth = F[F.family.eq('persistence')][KEY + ['age_type', 'Q_true']]  # float64 truth from the data table
        piv = piv.merge(truth, on=KEY, how='left', validate='one_to_one')
        assert piv[LEARNED + ['ARIMA_AICc', 'sqrt_efc_lag1', 'Q_true']].notna().all().all(), 'misaligned pivot'
        piv['best_NN'] = piv[best_nn]; piv['best_learned'] = piv[best_l]
        combos = {'C_arima_xgb_nn': ['ARIMA_AICc', 'xgb', 'best_NN'], 'C_arima_xgb': ['ARIMA_AICc', 'xgb'], 'C_arima_nn': ['ARIMA_AICc', 'best_NN'],
                  'C_xgb_nn': ['xgb', 'best_NN'], 'C_sqrt_xgb_nn': ['sqrt_efc_lag1', 'xgb', 'best_NN'],
                  'C_sqrt_bestlearned': ['sqrt_efc_lag1', 'best_learned'], 'C_all_learned': LEARNED}
        frames = []
        for name, members in [('best_NN', ['best_NN']), ('best_learned', ['best_learned'])] + list(combos.items()):
            g = piv[KEY + ['step', 'age_type', 'role', 'Q_true']].copy(); g['Q_pred'] = piv[members].mean(axis=1, skipna=False); g['family'] = name; g['fold'] = fold
            g['chosen'] = best_nn if name == 'best_NN' else best_l if name == 'best_learned' else ''
            frames.append(g)
        out.append(pd.concat(frames, ignore_index=True).assign(ape=lambda x: 100 * (x.Q_pred - x.Q_true).abs() / x.Q_true))
    allp = pd.concat(out, ignore_index=True)
    # selector
    sel_rows, picks = [], []
    cands = [m for m in allp.family.unique() if m not in ('best_NN', 'best_learned')]
    for fold in range(1, 6):
        F = allp[allp.fold.eq(fold)]; val = F[F.role.eq('validation')]
        overall = min(cands, key=lambda m: ecm(val[val.family.eq(m)]))
        for chem in (1, 2, 3):
            vc = val[val.age_type.eq(chem)]
            pick = min(cands, key=lambda m: ecm(vc[vc.family.eq(m)])) if vc.cell_id.nunique() >= 2 else overall
            picks.append(dict(fold=fold, chemistry=CHEMNAME[chem], validation_cells=vc.cell_id.nunique(), selected=pick))
            t = F[(F.age_type.eq(chem)) & F.family.eq(pick)].copy(); t['family'] = 'selector_by_chemistry'; t['chosen'] = pick; sel_rows.append(t)
    pd.DataFrame(picks).to_csv(OUT / 'selector_picks.csv', index=False)
    return pd.concat([allp] + sel_rows, ignore_index=True)


def bucket(step):
    return np.select([step <= 3, step <= 6, step <= 12], [step, 4, 7], 13)


def conformal(test):
    test = test.copy(); test['res'] = (test.Q_pred - test.Q_true).abs(); test['b'] = bucket(test.step)
    rows = []
    for (m, fold), T in test.groupby(['family', 'fold']):
        other = test[(test.family.eq(m)) & (test.fold.ne(fold))]
        for a in (.2, .1):
            q = {}
            for b, r in other.groupby('b').res:
                r = np.sort(r.to_numpy()); n = len(r); k = int(np.ceil((n + 1) * (1 - a)))
                q[b] = r[k - 1] if k <= n else np.inf
            half = T.b.map(q).fillna(np.inf)
            cov = (T.res <= half)
            rows.append(pd.DataFrame(dict(family=m, fold=fold, nominal=1 - a, cell_id=T.cell_id, age_type=T.age_type, seed=T.seed, covered=cov.to_numpy(), width_pp=100 * 2 * half.to_numpy())))
    c = pd.concat(rows, ignore_index=True)
    per_cell = c.groupby(['family', 'nominal', 'age_type', 'cell_id']).agg(cov=('covered', 'mean'), width=('width_pp', 'mean')).reset_index()
    overall = per_cell.groupby(['family', 'nominal']).agg(coverage_equal_cell=('cov', 'mean'), mean_width_SOH_pp=('width', 'mean')).reset_index()
    by_type = per_cell.assign(chem=per_cell.age_type.map(CHEMNAME)).groupby(['family', 'nominal', 'chem'])['cov'].mean().unstack('chem').reset_index()
    return overall.merge(by_type, on=['family', 'nominal'])


def decisions(test):
    g = test.groupby(['family', 'cell_id', 'seed'])
    rows = []
    for (m, cid, sd), x in g:
        x = x.sort_values('step'); last = x.iloc[-1]
        tc = x.step[x.Q_true <= .8]; pc = x.step[x.Q_pred <= .8]
        rows.append(dict(family=m, cell_id=cid, seed=sd, last_ape=100 * abs(last.Q_pred - last.Q_true) / last.Q_true,
                         last_signed=100 * (last.Q_pred - last.Q_true) / last.Q_true, true_cross=len(tc) > 0, pred_cross=len(pc) > 0,
                         cross_err=(pc.iloc[0] - tc.iloc[0]) if len(tc) and len(pc) else np.nan))
    d = pd.DataFrame(rows)
    out = []
    for m, x in d.groupby('family'):
        f1s = []
        for sd, y in x.groupby('seed'):
            tp = (y.true_cross & y.pred_cross).sum(); fp = (~y.true_cross & y.pred_cross).sum(); fn = (y.true_cross & ~y.pred_cross).sum()
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
        cell = x.groupby('cell_id')[['last_ape', 'last_signed']].mean()
        out.append(dict(family=m, last_step_APE=cell.last_ape.mean(), last_step_signed_pct=cell.last_signed.mean(),
                        share_optimistic=(cell.last_signed > 0).mean(), crossing_F1=np.nanmean(f1s),
                        true_crossing_cells=int(x[x.seed == x.seed.iloc[0]].true_cross.sum()),
                        mean_abs_crossing_step_error=x.cross_err.abs().mean()))
    return pd.DataFrame(out).sort_values('last_step_APE')


def boot(x):
    b = np.random.default_rng(20261005).choice(x, (10000, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); r = 0
    for i, j in enumerate(o):
        r = max(r, (len(p) - i) * p[j]); adj[j] = min(1, r)
    return adj


def main():
    allp, _ = load(); allp = add_derived(allp)
    test = allp[allp.role.eq('test')].copy()
    cell = test.groupby(['family', 'seed', 'cell_id']).ape.mean().groupby(['family', 'cell_id']).mean().unstack(0)
    types = test.groupby('cell_id').age_type.first().map(CHEMNAME)
    summ = []
    for m in cell.columns:
        x = cell[m].dropna(); lo, hi = boot(x.to_numpy())
        summ.append(dict(method=m, n_cells=len(x), equal_cell_MAPE=x.mean(), ci95_low=lo, ci95_high=hi, **{f'MAPE_{t}': cell.loc[types == t, m].mean() for t in ('LFP', 'NCA', 'NMC')}))
    summ = pd.DataFrame(summ).sort_values('equal_cell_MAPE'); summ.to_csv(OUT / 'extras_summary.csv', index=False)
    derived = ['selector_by_chemistry', 'best_NN', 'best_learned'] + [c for c in cell.columns if c.startswith('C_')] + ([c for c in cell.columns if c.startswith('Chronos')])
    fams = []
    for ref in ('ARIMA_AICc', 'sqrt_efc_lag1', 'rf'):
        rows = []
        for m in derived:
            if m == ref or m not in cell:
                continue
            d = (cell[m] - cell[ref]).dropna(); lo, hi = boot(d.to_numpy())
            rows.append(dict(family=f'snl_o{S.ORIGIN}_extras_vs_{ref}', comparison=f'{m} - {ref}', n=len(d), mean_diff=d.mean(), ci95_low=lo, ci95_high=hi,
                             p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
        r = pd.DataFrame(rows); r['p_holm'] = holm(r.p); fams.append(r)
    pairs = pd.concat(fams); pairs.to_csv(OUT / 'extras_paired_tests.csv', index=False)
    conf = conformal(test); conf.to_csv(OUT / 'conformal_summary.csv', index=False)
    dec = decisions(test); dec.to_csv(OUT / 'decision_metrics.csv', index=False)
    ch = OUT / 'chronos_pred.csv'
    if ch.exists():
        q = pd.read_csv(ch); q['cov80'] = (q.Q_true >= q.q10) & (q.Q_true <= q.q90); q['w'] = 100 * (q.q90 - q.q10)
        pc = q.groupby('cell_id')[['cov80', 'w']].mean(); pd.DataFrame([dict(method='ChronosBolt_small native q10-q90', coverage_equal_cell=pc.cov80.mean(), mean_width_SOH_pp=pc.w.mean())]).to_csv(OUT / 'chronos_native_coverage.csv', index=False)
    pd.set_option('display.width', 220)
    print(summ.round(3).to_string(index=False)); print(pairs.round(4).to_string(index=False))
    print(conf[conf.family.isin(['rf', 'xgb', 'compact_lstm', 'best_learned', 'sqrt_efc_lag1', 'ARIMA_AICc', 'C_sqrt_bestlearned', 'C_all_learned', 'selector_by_chemistry'] + [c for c in conf.family.unique() if c.startswith('Chronos')])].round(3).to_string(index=False))
    print(dec.round(3).to_string(index=False)); print(pd.read_csv(OUT / 'selector_picks.csv').to_string(index=False))


if __name__ == '__main__':
    main()
