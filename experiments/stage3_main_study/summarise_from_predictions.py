"""Recompute the main-study summary tables from the deposited prediction files (Level-1 reproduction).

Added for the public repository. The summary blocks of r4_transfer.py, r4_uncert.luh_intervals, r4_g.py (g1, g4)
and r4_ablate.py sit after model training in those scripts, so they are re-implemented here with the same
definitions, reading the saved predictions instead of retraining. It also regenerates the post hoc tables that were
produced interactively during the study (error by horizon, error by ageing regime, post hoc paired tests).

Definitions (manuscript Section 3.5): absolute percentage error per future checkup; mean over steps within a cell,
then over seeds, then over cells (equal-cell MAPE). Paired two-sided Wilcoxon signed-rank tests on cell-level
errors; Holm adjustment within the stated family.

Usage:  python summarise_from_predictions.py --pred ../../results/predictions --data ../../data/processed --out OUTDIR
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o):
        r = max(r, (len(p) - i) * p[j]); a[j] = min(1, r)
    return a


def equal_cell(df, keys, err='ape'):
    """Per (keys..., seed, cell) mean over steps -> mean over seeds -> Series indexed by keys + cell_id."""
    return df.groupby(keys + ['seed', 'cell_id'])[err].mean().groupby(level=keys + ['cell_id']).mean()


def read(pred, name):
    p = pred / f'{name}.csv.gz'
    return pd.read_csv(p if p.exists() else pred / f'{name}.csv')


def luh_test_frame(pred):
    """Test-cell predictions for the 104-cell cohort, on the Q scale, with the fixed ARIMA-residual average."""
    ref = read(pred, 'r3_reference_predictions'); ref = ref[ref.role.eq('test')]
    h = read(pred, 'r4_hybrid_pred'); h = h[h.role.eq('test')]
    cols = ['method', 'origin', 'seed', 'cell_id', 'row', 'step', 'Q_true', 'Q_pred']
    a = pd.concat([ref[cols], h[cols]], ignore_index=True)
    coh = set(ref.loc[ref.method.eq('ARIMA_AICc'), 'cell_id']); a = a[a.cell_id.isin(coh)]
    ar = a[a.method.eq('ARIMA_AICc')].drop(columns=['method', 'seed']).drop_duplicates(['origin', 'cell_id', 'row'])
    z = a[a.method.eq('hybrid_sqrt_xgb')].merge(ar[['origin', 'cell_id', 'row', 'Q_pred']], on=['origin', 'cell_id', 'row'], suffixes=('', '_a'), validate='many_to_one')
    z['Q_pred'] = (z.Q_pred + z.Q_pred_a) / 2; z['method'] = 'avg_ARIMA_hybrid'
    a = pd.concat([a, z[cols]], ignore_index=True); a['ape'] = 100 * (a.Q_pred - a.Q_true).abs() / a.Q_true
    return a


def _best_neural(pred, a):
    """r4_best_neural: per (origin, fold) the recurrent family with the lowest mean validation error (as in r4_analyse.py)."""
    f = read(pred, 'r4_final_pred'); v = read(pred, 'r4_final_runs'); v = v[v.family.isin(['compact_lstm', 'gru', 'large_lstm'])]
    b = v.groupby(['origin', 'fold', 'family']).val_Q_MAPE_equal_cell.mean().reset_index().sort_values('val_Q_MAPE_equal_cell').groupby(['origin', 'fold']).head(1)
    y = f[f.role.eq('test')].merge(b[['origin', 'fold', 'family']], on=['origin', 'fold', 'family']); y = y[y.cell_id.isin(set(a.cell_id))].copy()
    y['method'] = 'r4_best_neural'; y['ape'] = 100 * (y.Q_pred - y.Q_true).abs() / y.Q_true
    return y[['method', 'origin', 'seed', 'cell_id', 'row', 'step', 'Q_true', 'Q_pred', 'ape']]


def horizon_and_regime(pred, data, out):
    a = luh_test_frame(pred)
    m3 = ['ARIMA_AICc', 'avg_ARIMA_hybrid', 'hybrid_sqrt_xgb']
    x = a[a.method.isin(m3)]
    REG = ['ARIMA_AICc', 'avg_ARIMA_XGB', 'avg_ARIMA_hybrid', 'hybrid_sqrt_xgb', 'r4_best_neural', 'regime_origin_selector', 'sqrt_k_lag1']
    # error by future step: mean over seeds per (cell, step), then over cells; n_cells = cells with that step
    cs = x.groupby(['method', 'origin', 'cell_id', 'step']).ape.mean()
    hz = cs.groupby(['origin', 'step', 'method']).mean().unstack('method').round(2).reset_index()
    hz.columns.name = None; hz.to_csv(out / 'r4_hybrid_error_by_horizon.csv', index=False)
    # error by ageing regime (age_type 1 calendar, 2 cyclic, 3 profile)
    d = pd.read_csv(data / 'luh_blank' / 'phase1_cv_all_rows.csv')
    at = d.groupby('cell_id').age_type.first()
    xr = pd.concat([a[a.method.isin(REG)], _best_neural(pred, a)], ignore_index=True)
    c = equal_cell(xr, ['method', 'origin']).reset_index(); c['age'] = c.cell_id.map(at)
    bt = c.groupby(['method', 'origin', 'age']).ape.mean().unstack(['origin', 'age']).round(2)
    bt.to_csv(out / 'r4_by_age_type.csv')
    return a


def posthoc(pred, a, out):
    cell = equal_cell(a, ['method', 'origin'])
    rows = []
    for o in (10, 15, 20):
        c = cell.xs(o, level='origin').unstack(0); fam = []
        for ref in ['sqrt_k_lag1', 'rf', 'xgb', 'best_neural', 'avg_ARIMA_XGB_NN', 'regime_origin_selector']:
            d = (c['hybrid_sqrt_xgb'] - c[ref]).dropna()
            fam.append(dict(origin=o, comparison=f'hybrid_sqrt_xgb - {ref}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        rows.append(pd.DataFrame(fam))
    r = pd.concat(rows, ignore_index=True); r['p_holm'] = holm(r.p)  # one Holm family over all 18 post hoc tests
    r.to_csv(out / 'r4_hybrid_vs_round3_posthoc.csv', index=False)


def tuned_posthoc(pred, out):
    """Hybrid vs origin-specifically re-tuned learned models (origins 10 and 15); one Holm family of 8."""
    ref = read(pred, 'r3_reference_predictions'); coh = set(ref.loc[ref.method.eq('ARIMA_AICc') & ref.role.eq('test'), 'cell_id'])
    h = read(pred, 'r4_hybrid_pred'); h = h[h.role.eq('test')].assign(method='hybrid_sqrt_xgb')
    f = read(pred, 'r4_final_pred'); f = f[f.role.eq('test')].copy(); f['method'] = 'r4_' + f.family
    v = read(pred, 'r4_final_runs'); v = v[v.family.isin(['compact_lstm', 'gru', 'large_lstm'])]
    b = v.groupby(['origin', 'fold', 'family']).val_Q_MAPE_equal_cell.mean().reset_index().sort_values('val_Q_MAPE_equal_cell').groupby(['origin', 'fold']).head(1)
    y = read(pred, 'r4_final_pred'); y = y[y.role.eq('test')].merge(b[['origin', 'fold', 'family']], on=['origin', 'fold', 'family']); y['method'] = 'r4_best_neural'
    cols = ['method', 'origin', 'seed', 'cell_id', 'row', 'Q_true', 'Q_pred']
    a = pd.concat([h[cols], f[cols], y[cols]]); a = a[a.cell_id.isin(coh)]
    ar = ref[ref.role.eq('test') & ref.method.eq('ARIMA_AICc')][['origin', 'cell_id', 'row', 'Q_pred']].drop_duplicates(['origin', 'cell_id', 'row'])
    for name, mem in (('avg_ARIMA_hybrid', 'hybrid_sqrt_xgb'), ('avg_ARIMA_r4NN', 'r4_best_neural')):
        z = a[a.method.eq(mem)].merge(ar, on=['origin', 'cell_id', 'row'], suffixes=('', '_a'), validate='many_to_one')
        z['Q_pred'] = (z.Q_pred + z.Q_pred_a) / 2; z['method'] = name; a = pd.concat([a, z[cols]], ignore_index=True)
    a['ape'] = 100 * (a.Q_pred - a.Q_true).abs() / a.Q_true
    cell = equal_cell(a, ['method', 'origin']); rows = []
    for o in (10, 15):
        c = cell.xs(o, level='origin').unstack(0)
        for m, ref in [('hybrid_sqrt_xgb', 'r4_best_neural'), ('hybrid_sqrt_xgb', 'r4_large_lstm'), ('hybrid_sqrt_xgb', 'r4_xgb'), ('avg_ARIMA_hybrid', 'avg_ARIMA_r4NN')]:
            d = (c[m] - c[ref]).dropna(); rows.append(dict(origin=o, comparison=f'{m} - {ref}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
    r = pd.DataFrame(rows); r['p_holm'] = holm(r.p); r.to_csv(out / 'r4_hybrid_vs_tuned_learned_posthoc.csv', index=False)


def transfer_kirkaldy(pred, out):
    R = read(pred, 'r4_transfer_kirkaldy_pred')
    cell = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean()
    S, P = [], []
    for o in (3, 5, 8):
        c = cell.xs(o, level='origin').unstack(0)
        for m in c.columns: S.append(dict(origin=o, method=m, n_cells=c[m].notna().sum(), MAPE=c[m].mean()))
        fam = []
        for ref in ('sqrt_idx_prior', 'ARIMA_AICc', 'persistence'):
            d = (c['hybrid_transfer'] - c[ref]).dropna(); fam.append(dict(origin=o, comparison=f'hybrid_transfer - {ref}', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        fam = pd.DataFrame(fam); fam['p_holm'] = holm(fam.p); P.append(fam)
    pd.DataFrame(S).to_csv(out / 'r4_transfer_summary.csv', index=False); pd.concat(P).to_csv(out / 'r4_transfer_pairs.csv', index=False)


def oxford(pred, out):
    R = read(pred / 'oxford', 'r4_g1_oxford_pred'); R['ape'] = 100 * (R.pred - R.soh_true).abs() / R.soh_true
    S = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean().groupby(['origin', 'method']).mean().unstack(0)
    S.to_csv(out / 'r4_g1_oxford_summary.csv')
    cell = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean(); P = []
    for o in (5, 10, 20):
        c = cell.xs(o, level='origin').unstack(0)
        for ref in ('ARIMA_AICc', 'sqrt_idx_prior', 'persistence'):
            d = c.hybrid_frozen - c[ref]; P.append(dict(origin=o, comparison=f'hybrid_frozen - {ref}', n=len(d), diff=d.mean(), wins=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue))
    pd.DataFrame(P).to_csv(out / 'r4_g1_oxford_pairs.csv', index=False)


def conformal_luh(pred, out):
    """Same procedure as r4_uncert.luh_intervals after its model fits (split conformal, per fold/origin/seed/step)."""
    P = read(pred, 'r4_hybrid_fitonly_pred'); rows = []
    for (fold, o, sd), g in P.groupby(['fold', 'origin', 'seed']):
        cal = g[g.role.eq('calibration')]; te = g[g.role.eq('test')].copy(); te['res'] = (te.Q_pred - te.Q_true).abs()
        for q in (.8, .9):
            rad = {}
            for st, r in cal.assign(res=(cal.Q_pred - cal.Q_true).abs()).groupby('step').res:
                r = np.sort(r.to_numpy()); n = len(r); k = int(np.ceil((n + 1) * q)); rad[st] = r[k - 1] if k <= n else np.inf
            hh = te.step.map(rad).fillna(np.inf)
            rows.append(pd.DataFrame(dict(fold=fold, origin=o, seed=sd, nominal=q, cell_id=te.cell_id, covered=(te.res <= hh).to_numpy(), width_Ah=2 * hh.to_numpy(), width_SOH=2 * hh.to_numpy() / te.Q0.to_numpy())))
    R = pd.concat(rows); R['finite'] = np.isfinite(R.width_Ah)
    c = R.groupby(['origin', 'nominal', 'cell_id']).agg(cov=('covered', 'mean'), wA=('width_Ah', lambda x: x[np.isfinite(x)].mean()), wS=('width_SOH', lambda x: x[np.isfinite(x)].mean()), fin=('finite', 'mean')).reset_index()
    S = c.groupby(['origin', 'nominal']).agg(coverage_equal_cell=('cov', 'mean'), mean_width_Ah=('wA', 'mean'), mean_width_SOH=('wS', 'mean'), finite_fraction=('fin', 'mean')).reset_index()
    t = P[P.role.eq('test')]; ape = 100 * (t.Q_pred - t.Q_true).abs() / t.Q_true
    acc = ape.groupby([t.origin, t.seed, t.cell_id]).mean().groupby(level=[0, 2]).mean().groupby(level=0).mean()
    S['fitonly_point_MAPE'] = S.origin.map(acc); S.to_csv(out / 'r4_hybrid_conformal_luh.csv', index=False)


def robustness(pred, out):
    R = read(pred, 'r4_g4_robust_pred')
    hc = R.groupby(['sigma', 'drop', 'rep', 'seed', 'cell_id']).ape.mean().groupby(['sigma', 'drop', 'cell_id']).mean().rename('hybrid')
    rc = read(pred, 'robustness_cell_errors').rename(columns={'noise_sigma_Q0': 'sigma', 'drop_fraction': 'drop'})
    piv = rc.pivot_table(index=['sigma', 'drop', 'cell_id'], columns='method', values='APE').join(hc)
    piv.groupby(level=[0, 1]).mean().to_csv(out / 'r4_g4_robust_summary.csv')
    P = []
    for (sg, dr), x in piv.groupby(level=[0, 1]):
        dd = (x.hybrid - x.ARIMA_AICc).dropna(); P.append(dict(sigma=sg, drop=dr, n=len(dd), hybrid_minus_ARIMA=dd.mean(), p=stats.wilcoxon(dd).pvalue))
    P = pd.DataFrame(P); P['p_holm'] = holm(P.p); P.to_csv(out / 'r4_g4_robust_pairs.csv', index=False)


def ablation_luh(pred, out):
    P = read(pred, 'r4_ablation_luh_pred'); P['ape'] = 100 * (P.pred - P.soh_true).abs() / P.soh_true
    s = P.groupby(['variant', 'origin', 'seed', 'cell_id']).ape.mean().groupby(['variant', 'origin', 'cell_id']).mean()
    s.groupby(['variant', 'origin']).mean().unstack().to_csv(out / 'r4_ablation_luh_summary.csv')
    rows = []
    for o in (10, 15, 20):
        c = s.xs(o, level='origin').unstack(0); fam = []
        for v in [x for x in c.columns if x != 'V1_main']:
            d = (c[v] - c['V1_main']).dropna(); fam.append(dict(dataset='luh', origin=o, comparison=f'{v} - V1_main', n=len(d), diff_pp=d.mean(), p=stats.wilcoxon(d).pvalue))
        fam = pd.DataFrame(fam); fam['p_holm'] = holm(fam.p); rows.append(fam)
    pd.concat(rows).to_csv(out / 'r4_ablation_luh_pairs.csv', index=False)


def noschedule_luh(pred, out):
    g3 = read(pred, 'r4_g3_noschedule_pred_luh'); g3['ape'] = 100 * (g3.pred - g3.soh_true).abs() / g3.soh_true
    h = read(pred, 'r4_hybrid_pred'); h = h[h.role.eq('test')].copy(); h['ape'] = 100 * (h.pred - h.soh_true).abs() / h.soh_true
    ref = read(pred, 'r3_reference_predictions'); coh = set(ref.loc[ref.method.eq('ARIMA_AICc') & ref.role.eq('test'), 'cell_id'])
    a = equal_cell(g3[g3.cell_id.isin(coh)], ['origin']); b = equal_cell(h[h.cell_id.isin(coh)], ['origin']); rows = []
    for o in (10, 15, 20):
        d = (a.xs(o) - b.xs(o)).dropna(); rows.append(dict(dataset='luh', origin=o, n=len(d), diff_noschedule_minus_main=d.mean(), p=stats.wilcoxon(d).pvalue, MAPE_noschedule=a.xs(o).mean(), MAPE_main=b.xs(o).mean()))
    pd.DataFrame(rows).to_csv(out / 'r4_g3_vs_main_pairs_luh.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    A = ap.parse_args(); pred, data, out = Path(A.pred), Path(A.data), Path(A.out); out.mkdir(parents=True, exist_ok=True)
    a = horizon_and_regime(pred, data, out); posthoc(pred, a, out); tuned_posthoc(pred, out)
    transfer_kirkaldy(pred, out); oxford(pred, out); conformal_luh(pred, out); robustness(pred, out); ablation_luh(pred, out); noschedule_luh(pred, out)
    print('summary tables written to', out)
