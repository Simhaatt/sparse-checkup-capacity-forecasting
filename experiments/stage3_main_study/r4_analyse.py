"""Round 4 analysis on Luh-Blank test cells (104) at origins 10/15/20 + Sandia hybrid/pooled summaries."""
from r3 import *
from scipy import stats
REF = ROOT / 'inputs/r3_reference_predictions.csv'


def boot(x):
    b = np.random.default_rng(20261005).choice(x, (10000, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o): r = max(r, (len(p) - i) * p[j]); a[j] = min(1, r)
    return a


def load():
    ref = pd.read_csv(REF); ref = ref[ref.role.eq('test')][['method', 'origin', 'seed', 'cell_id', 'row', 'Q_true', 'Q_pred']]
    fr = [ref]
    h = ROOT / 'results/r4_hybrid_pred.csv'
    if h.exists():
        x = pd.read_csv(h); x = x[x.role.eq('test')]; fr.append(x[['method', 'origin', 'seed', 'cell_id', 'row', 'Q_true', 'Q_pred']])
    f = ROOT / 'results/r4_final_pred.csv'
    if f.exists():
        x = pd.read_csv(f); x = x[x.role.eq('test')].copy(); x['method'] = 'r4_' + x.family; fr.append(x[['method', 'origin', 'seed', 'cell_id', 'row', 'Q_true', 'Q_pred']])
        # best neural per fold/origin by mean validation of finals
        v = pd.read_csv(ROOT / 'results/r4_final_runs.csv'); v = v[v.family.isin(['compact_lstm', 'gru', 'large_lstm'])]
        b = v.groupby(['origin', 'fold', 'family']).val_Q_MAPE_equal_cell.mean().reset_index().sort_values('val_Q_MAPE_equal_cell').groupby(['origin', 'fold']).head(1)
        y = pd.read_csv(f); y = y[y.role.eq('test')].merge(b[['origin', 'fold', 'family']], on=['origin', 'fold', 'family']); y['method'] = 'r4_best_neural'
        fr.append(y[['method', 'origin', 'seed', 'cell_id', 'row', 'Q_true', 'Q_pred']])
    a = pd.concat(fr, ignore_index=True)
    coh = set(ref.loc[ref.method.eq('ARIMA_AICc'), 'cell_id']); a = a[a.cell_id.isin(coh)]  # 104-cell long cohort for every method
    ar = a[a.method.eq('ARIMA_AICc')].drop(columns=['method', 'seed']).drop_duplicates(['origin', 'cell_id', 'row'])
    out = [a]
    for name, mem in (('avg_ARIMA_hybrid', 'hybrid_sqrt_xgb'), ('avg_ARIMA_r4xgb', 'r4_xgb'), ('avg_ARIMA_r4NN', 'r4_best_neural')):
        m = a[a.method.eq(mem)]
        if len(m):
            z = m.merge(ar[['origin', 'cell_id', 'row', 'Q_pred']], on=['origin', 'cell_id', 'row'], suffixes=('', '_a'), validate='many_to_one')
            z['Q_pred'] = (z.Q_pred + z.Q_pred_a) / 2; z['method'] = name; out.append(z[a.columns])
    a = pd.concat(out, ignore_index=True); a['ape'] = 100 * (a.Q_pred - a.Q_true).abs() / a.Q_true
    return a


def main():
    a = load(); cell = a.groupby(['method', 'origin', 'seed', 'cell_id']).ape.mean().groupby(['method', 'origin', 'cell_id']).mean()
    rows, pairs = [], []
    for o in (10, 15, 20):
        c = cell.xs(o, level='origin').unstack(0)
        for m in c.columns:
            x = c[m].dropna(); lo, hi = boot(x.to_numpy()); rows.append(dict(origin=o, method=m, n_cells=len(x), Q_MAPE=x.mean(), ci95_low=lo, ci95_high=hi))
        new = [m for m in c.columns if m.startswith(('r4_', 'hybrid', 'avg_ARIMA_hybrid', 'avg_ARIMA_r4'))]
        for ref, fam in (('ARIMA_AICc', f'r4_vs_ARIMA_origin{o}'), ('avg_ARIMA_XGB', f'r4_vs_round3best_origin{o}')):
            fr = []
            for m in new:
                d = (c[m] - c[ref]).dropna(); lo, hi = boot(d.to_numpy())
                fr.append(dict(family=fam, origin=o, comparison=f'{m} - {ref}', n=len(d), diff_pp=d.mean(), ci95_low=lo, ci95_high=hi, p=stats.wilcoxon(d).pvalue))
            if fr:
                fr = pd.DataFrame(fr); fr['p_holm'] = holm(fr.p); pairs.append(fr)
    S = pd.DataFrame(rows); S.to_csv(ROOT / 'tables/r4_summary.csv', index=False)
    P = pd.concat(pairs) if pairs else pd.DataFrame(); P.to_csv(ROOT / 'tables/r4_pairs.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300)
    print(S.pivot(index='method', columns='origin', values='Q_MAPE').round(3).sort_values(20).to_string()); print(P.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
