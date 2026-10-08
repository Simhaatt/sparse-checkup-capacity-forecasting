"""Build the verified plotting tables in data_used/ from the round-4 result files.

Run once in the analysis workspace (needs the round-4 tables/ and results/ folders):
    python prepare_plot_data.py --r4 /path/to/r4 --out ../data_used
Every summary value is copied from the round-4 tables. The only derived quantities are
(i) 95 % cell-bootstrap CIs for fine-tuned Chronos (Fig. 3b) and the transfer bars (Fig. 5),
computed with the same procedure as r4_analyse.boot (10,000 resamples of cells, numpy
default_rng(20261005), percentile interval), and (ii) the number of cells contributing to
each forecast step (Fig. 4). Each derived mean is asserted equal to the published summary.
"""
import argparse, shutil, numpy as np, pandas as pd
from pathlib import Path


def boot(x):
    b = np.random.default_rng(20261005).choice(x, (10000, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def cell_mape(df, true='soh_true', pred='pred'):
    ape = 100 * (df[pred] - df[true]).abs() / df[true]
    return ape.groupby([df.seed, df.cell_id]).mean().groupby(level=1).mean()


def main(r4, out):
    T, R = r4 / 'tables', r4 / 'results'; out.mkdir(parents=True, exist_ok=True); src = out / 'source_tables'; src.mkdir(exist_ok=True)
    for f in ['r4_summary.csv', 'r4_pairs.csv', 'r4_chronos_ft_summary.csv', 'r4_chronos_ft_pairs.csv', 'r4_hybrid_error_by_horizon.csv',
              'r4_transfer_summary.csv', 'r4_transfer_pairs.csv', 'r4_g1_oxford_summary.csv', 'r4_g1_oxford_pairs.csv', 'r4_hybrid_conformal_luh.csv',
              'r4_g5_mondrian_by_regime.csv', 'r4_hybrid_conformal_snl.csv', 'r4_decisions_luh_r3rule.csv']:
        shutil.copy(T / f, src / f)

    # ---- Fig. 3: accuracy with CIs ----
    s = pd.read_csv(T / 'r4_summary.csv')
    keep = {'sqrt_k_lag1': 'Square-root prior', 'ARIMA_AICc': 'ARIMA', 'hybrid_sqrt_xgb': 'Residual learner', 'avg_ARIMA_hybrid': 'ARIMA–residual average',
            'avg_ARIMA_XGB': 'ARIMA–XGBoost average (round 3)', 'r4_best_neural': 'Best neural, re-tuned per origin'}
    a = s[s.method.isin(keep)].rename(columns={'Q_MAPE': 'MAPE', 'ci95_low': 'ci_low', 'ci95_high': 'ci_high'})[['method', 'origin', 'n_cells', 'MAPE', 'ci_low', 'ci_high']]
    a['label'] = a.method.map(keep); a['ci_source'] = 'r4_summary.csv'
    c = pd.read_csv(R / 'r4_chronos_ft_luh_pred.csv'); c['seed'] = 0
    cs = pd.read_csv(T / 'r4_chronos_ft_summary.csv'); cs = cs[cs.dataset.eq('luh') & cs.method.eq('ChronosBolt_small_finetuned')].set_index('origin').MAPE
    rows = []
    for o, g in c.groupby('origin'):
        x = cell_mape(g).to_numpy(); lo, hi = boot(x)
        assert abs(x.mean() - cs[o]) < 1e-9, ('chronos mean mismatch', o)
        rows.append(dict(method='ChronosBolt_small_finetuned', origin=o, n_cells=len(x), MAPE=x.mean(), ci_low=lo, ci_high=hi,
                         label='Chronos-Bolt-small, fine-tuned', ci_source='bootstrap of r4_chronos_ft_luh_pred.csv (this script)'))
    a = pd.concat([a, pd.DataFrame(rows)], ignore_index=True).sort_values(['label', 'origin'])
    a.to_csv(out / 'fig3_primary_accuracy.csv', index=False)
    p = pd.read_csv(T / 'r4_pairs.csv')
    p = p[p.comparison.isin(['avg_ARIMA_hybrid - ARIMA_AICc', 'hybrid_sqrt_xgb - ARIMA_AICc'])][['family', 'origin', 'comparison', 'n', 'diff_pp', 'p', 'p_holm']]
    p.to_csv(out / 'fig3_holm_vs_arima.csv', index=False)

    # ---- Fig. 4: horizon curves (full precision, recomputed from saved predictions) + cells per step ----
    import sys; sys.path.insert(0, str(r4))
    import r4_analyse as A
    a = A.load(); a = a[a.method.isin(['ARIMA_AICc', 'hybrid_sqrt_xgb', 'avg_ARIMA_hybrid'])]
    a = a.merge(a.groupby(['origin', 'cell_id']).row.min().rename('r0').reset_index(), on=['origin', 'cell_id']); a['step'] = a.row - a.r0 + 1
    e = (a.groupby(['method', 'origin', 'step', 'seed', 'cell_id']).ape.mean().groupby(['method', 'origin', 'step', 'cell_id']).mean()
         .groupby(['method', 'origin', 'step']).mean().unstack('method').reset_index())
    n = a[a.method.eq('hybrid_sqrt_xgb')].groupby(['origin', 'step']).cell_id.nunique().rename('n_cells').reset_index()
    h = e.merge(n, on=['origin', 'step'])
    ref = pd.read_csv(T / 'r4_hybrid_error_by_horizon.csv').set_index(['origin', 'step'])
    for c in ['ARIMA_AICc', 'avg_ARIMA_hybrid', 'hybrid_sqrt_xgb']:
        assert (h.set_index(['origin', 'step'])[c].round(2) - ref[c]).abs().max() < 1e-9, c
    h.to_csv(out / 'fig4_error_by_horizon.csv', index=False)

    # ---- Fig. 5: transfer with CIs ----
    k = pd.read_csv(R / 'r4_transfer_kirkaldy_pred.csv'); ks = pd.read_csv(T / 'r4_transfer_summary.csv').set_index(['origin', 'method']).MAPE
    ox = pd.read_csv(R / 'r4_g1_oxford_pred.csv'); os_ = pd.read_csv(T / 'r4_g1_oxford_summary.csv').set_index('method')
    lab = {'ARIMA_AICc': 'ARIMA', 'hybrid_transfer': 'Residual learner (frozen, unit-free)', 'hybrid_frozen': 'Residual learner (frozen, unit-free)', 'sqrt_idx_prior': 'Square-root prior (index)'}
    rows = []
    for ds, df, ref in (('Kirkaldy 21700 (40 cells)', k, lambda o, m: ks[(o, m)]), ('Oxford pouch (8 cells)', ox, lambda o, m: os_.loc[m, str(o)])):
        for (o, m), g in df[df.method.isin(lab)].groupby(['origin', 'method']):
            x = cell_mape(g).to_numpy(); lo, hi = boot(x)
            assert abs(x.mean() - ref(o, m)) < 1e-9, (ds, o, m, x.mean(), ref(o, m))
            rows.append(dict(dataset=ds, origin=o, method=m, label=lab[m], n_cells=len(x), MAPE=x.mean(), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(out / 'fig5_transfer.csv', index=False)

    # ---- Fig. 6: copies of verified tables ----
    for f in ['r4_hybrid_conformal_luh.csv', 'r4_g5_mondrian_by_regime.csv', 'r4_hybrid_conformal_snl.csv', 'r4_decisions_luh_r3rule.csv']:
        shutil.copy(T / f, out / ('fig6_' + f))
    # NCA cell counts behind panel (c)
    sp = pd.read_csv(R / 'r4_hybrid_snl_pred.csv'); sp = sp[sp.role.eq('test')]; sp['chem'] = sp.cell_id.str.extract(r'_(LFP|NCA|NMC)_')
    sp.groupby(['origin', 'chem']).cell_id.nunique().rename('n_cells').reset_index().to_csv(out / 'fig6_sandia_cells_by_chemistry.csv', index=False)
    print('written to', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--r4', default=str(Path(__file__).resolve().parents[2] / 'experiments' / 'stage3_main_study')); ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'data_used'))  # repo edit: --r4 default was '/home/claude/r4'
    a = ap.parse_args(); main(Path(a.r4), Path(a.out))
