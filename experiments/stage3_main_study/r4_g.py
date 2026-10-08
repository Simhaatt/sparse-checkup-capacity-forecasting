"""R4G reviewer-robustness studies. Usage: python r4_g.py g1|g2|g3|g4|g5|g6"""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'trainer')); sys.path.insert(0, _SNL)
from xgboost import XGBRegressor
SEEDS = [42, 52, 62]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o): r = max(r, (len(p) - i) * p[j]); a[j] = min(1, r)
    return a


def eq(df, col='pred', truth='soh_true'):
    return (100 * (df[col] - df[truth]).abs() / df[truth]).groupby([df.seed, df.cell_id]).mean().groupby(level=1).mean()


# ---------------- G1 Oxford frozen confirmation + long horizon
def g1():
    import r4_pooled as P, r4_transfer as TR
    from univariate import arima_prefix
    src = pd.concat([P.luh(), P.snl()], ignore_index=True)[P.COLS]
    T = pd.DataFrame([r for _, g in src.groupby('cell_id') for r in TR.rows(g, range(3, 80))])
    ox = P.oxf()[P.COLS]; res = []
    for o in (5, 10, 20):
        E = pd.DataFrame([r for _, g in ox.groupby('cell_id') for r in TR.rows(g, [o])])
        for sd in SEEDS:
            m = XGBRegressor(**TR.CFG, random_state=sd).fit(T[TR.FEAT], T.soh_true - T.prior)
            res.append(E.assign(method='hybrid_frozen', seed=sd, pred=np.maximum(E.prior + m.predict(E[TR.FEAT]), 1e-6)))
        res.append(E.assign(method='sqrt_idx_prior', seed=0, pred=E.prior)); res.append(E.assign(method='persistence', seed=0, pred=E.last_obs))
        ar = []
        for cid, g in ox.groupby('cell_id'):
            g = g.sort_values('timestamp_s'); y = (g.Q / g.Q.iloc[0]).to_numpy(); f, _ = arima_prefix(y[:o], len(y) - o)
            ar.append(pd.DataFrame(dict(cell_id=cid, row=np.arange(o, len(y)), pred=f)))
        res.append(E[['cell_id', 'row', 'step', 'origin', 'soh_true']].merge(pd.concat(ar), on=['cell_id', 'row']).assign(method='ARIMA_AICc', seed=0))
    R = pd.concat(res, ignore_index=True); R.to_csv(ROOT / 'results/r4_g1_oxford_pred.csv', index=False)
    R['ape'] = 100 * (R.pred - R.soh_true).abs() / R.soh_true
    R['hbin'] = pd.cut(R.step, [0, 5, 15, 30, 1000], labels=['1-5', '6-15', '16-30', '31+'])
    S = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean().groupby(['origin', 'method']).mean().unstack(0)
    Hb = R.groupby(['origin', 'hbin', 'method'], observed=True).ape.mean().unstack('method')
    cell = R.groupby(['origin', 'method', 'seed', 'cell_id']).ape.mean().groupby(['origin', 'method', 'cell_id']).mean(); pairs = []
    for o in (5, 10, 20):
        c = cell.xs(o, level='origin').unstack(0)
        for ref in ('ARIMA_AICc', 'sqrt_idx_prior', 'persistence'):
            d = c.hybrid_frozen - c[ref]; pairs.append(dict(origin=o, comparison=f'hybrid_frozen - {ref}', n=len(d), diff=d.mean(), wins=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue))
    Pp = pd.DataFrame(pairs); S.to_csv(ROOT / 'tables/r4_g1_oxford_summary.csv'); Hb.to_csv(ROOT / 'tables/r4_g1_oxford_by_horizon.csv'); Pp.to_csv(ROOT / 'tables/r4_g1_oxford_pairs.csv', index=False)
    print(S.round(3).to_string()); print(Hb.round(3).to_string()); print(Pp.round(4).to_string(index=False))


# ---------------- G2 knee subsets
def knee_flags(seqs):
    out = {}
    for cid, y in seqs.items():
        n = len(y)
        if n < 6: out[cid] = False; continue
        t = max(2, n // 3)
        r1 = (y[0] - y[t]) / t; r3 = (y[n - 1 - t] - y[n - 1]) / t
        out[cid] = bool(r3 > 2 * max(r1, 1e-4))
    return out


def g2():
    import r3, r4_pooled as P
    rows = []
    # Luh
    d = r3.data(); seqs = {c: (g.sort_values('timestamp_s').Q / g.sort_values('timestamp_s').Q.iloc[0]).to_numpy() for c, g in d.groupby('cell_id')}; kn = knee_flags(seqs)
    ref = pd.read_csv(ROOT / 'inputs/r3_reference_predictions.csv'); ref = ref[ref.role.eq('test') & ref.method.isin(['ARIMA_AICc', 'sqrt_k_lag1', 'avg_ARIMA_XGB'])]
    ref = ref.assign(pred=ref.Q_pred / ref.Q0, soh_true=ref.Q_true / ref.Q0)
    h = pd.read_csv(ROOT / 'results/r4_hybrid_pred.csv'); h = h[h.role.eq('test')].assign(pred=lambda x: x.pred)
    for (m, o), x in pd.concat([ref[['method', 'origin', 'seed', 'cell_id', 'pred', 'soh_true']], h[['method', 'origin', 'seed', 'cell_id', 'pred', 'soh_true']]]).groupby(['method', 'origin']):
        e = eq(x); rows.append(dict(dataset='Luh', method=m, origin=o, knee_cells=int(sum(kn[c] for c in e.index)), MAPE_knee=e[[kn[c] for c in e.index]].mean(), MAPE_nonknee=e[[not kn[c] for c in e.index]].mean()))
    # Sandia
    import snl_bench as S
    sd = S.data(); seqs = {c: g.sort_values('timestamp_s').Q.to_numpy() for c, g in sd.groupby('cell_id')}; kn2 = knee_flags(seqs)
    hs = pd.read_csv(ROOT / 'results/r4_hybrid_snl_pred.csv'); hs = hs[hs.role.eq('test')].assign(method='hybrid_sqrt_xgb')
    for o in (5, 8):
        c = pd.read_csv(f'{_SNL}/origin{o}/controls_pred.csv'); c = c[c.family.isin(['ARIMA_AICc', 'sqrt_efc_lag1'])].rename(columns={'family': 'method', 'Q_true': 'soh_true', 'Q_pred': 'pred'}); c['origin'] = o
        for m, x in pd.concat([c[['method', 'origin', 'seed', 'cell_id', 'pred', 'soh_true']], hs[hs.origin.eq(o)][['method', 'origin', 'seed', 'cell_id', 'pred', 'soh_true']]]).groupby('method'):
            e = eq(x); rows.append(dict(dataset='Sandia', method=m, origin=o, knee_cells=int(sum(kn2[c_] for c_ in e.index)), MAPE_knee=e[[kn2[c_] for c_ in e.index]].mean(), MAPE_nonknee=e[[not kn2[c_] for c_ in e.index]].mean()))
    # Kirkaldy
    k = P.kir(); seqs = {c: g.sort_values('timestamp_s').Q.to_numpy() for c, g in k.groupby('cell_id')}; kn3 = knee_flags(seqs)
    t = pd.read_csv(ROOT / 'results/r4_transfer_kirkaldy_pred.csv')
    for (m, o), x in t.groupby(['method', 'origin']):
        e = eq(x); rows.append(dict(dataset='Kirkaldy', method=m, origin=o, knee_cells=int(sum(kn3[c_] for c_ in e.index)), MAPE_knee=e[[kn3[c_] for c_ in e.index]].mean(), MAPE_nonknee=e[[not kn3[c_] for c_ in e.index]].mean()))
    R = pd.DataFrame(rows); R.to_csv(ROOT / 'tables/r4_g2_knee.csv', index=False)
    print('knee cells: Luh', sum(kn.values()), '/', len(kn), 'Sandia', sum(kn2.values()), '/', len(kn2), 'Kirkaldy', sum(kn3.values()), '/', len(kn3)); print(R.round(3).to_string(index=False))


# ---------------- G3 no-schedule hybrid (index axis only)
STATIC = ['temperature', 'c_rate_chg', 'c_rate_dischg', 'soc_window', 'age_type']
def g3():
    import r3, r4_transfer as TR, r4_hybrid as H
    FEAT = TR.FEAT + STATIC
    def frame(d, origins):
        rows = []
        for cid, g in d.groupby('cell_id'):
            g = g.sort_values('timestamp_s', kind='mergesort').reset_index(drop=True)
            for r in TR.rows(g, origins):
                r.update({c: float(g[c].iloc[0]) for c in STATIC}); r['Q0'] = g.Q.iloc[0]; rows.append(r)
        return pd.DataFrame(rows)
    out = []
    for name in ('luh', 'snl'):
        if name == 'luh':
            d = r3.data(); long = r3.cohort(d); origins = (10, 15, 20); start = 8; stepk = 2
        else:
            import snl_bench as S
            d = S.data().copy(); d['Q'] = d.Q * d.Q0; origins = (5, 8); start = 3; stepk = 1
        for fold in range(1, 6):
            for o in origins:
                if name == 'luh':
                    s = r3.split(fold, d); tr = pd.concat([s['fit'], s['calibration']]); va = s['validation'][s['validation'].cell_id.isin(long)]; te = s['test'][s['test'].cell_id.isin(long)]
                else:
                    p = pd.read_csv(f'{_SNL}/origin{o}/partitions.csv'); p = p[p.fold.eq(fold)].set_index('cell_id').role; r = d.cell_id.map(p)
                    n = d.groupby('cell_id').size(); sc = set(n[n > o].index); tr = d[r.eq('fit')]; va = d[r.eq('validation') & d.cell_id.isin(sc)]; te = d[r.eq('test') & d.cell_id.isin(sc)]
                T = frame(tr, list(range(start, tr.groupby('cell_id').size().max() - 1, stepk))); V = frame(va, [o]); E = frame(te, [o]); sc_ = []
                for c in H.CONFIGS:
                    m = XGBRegressor(**c, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=42).fit(T[FEAT], T.soh_true - T.prior)
                    sc_.append(float((100 * (np.maximum(V.prior + m.predict(V[FEAT]), 1e-6) - V.soh_true).abs() / V.soh_true).groupby(V.cell_id).mean().mean()))
                b = int(np.argmin(sc_))
                for sd in SEEDS:
                    m = XGBRegressor(**H.CONFIGS[b], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[FEAT], T.soh_true - T.prior)
                    out.append(E[['cell_id', 'row', 'step', 'soh_true']].assign(pred=np.maximum(E.prior + m.predict(E[FEAT]), 1e-6), dataset=name, origin=o, fold=fold, seed=sd))
            print('g3', name, 'fold', fold, flush=True)
    R = pd.concat(out); R.to_csv(ROOT / 'results/r4_g3_noschedule_pred.csv', index=False)
    S = pd.DataFrame([dict(dataset=dn, origin=o, MAPE=eq(x).mean()) for (dn, o), x in R.groupby(['dataset', 'origin'])]); S.to_csv(ROOT / 'tables/r4_g3_noschedule_summary.csv', index=False); print(S.round(3).to_string(index=False))


# ---------------- G4 noise / dropout robustness at origin 20 (Luh)
def g4():
    import r3, r4_hybrid as H
    d = r3.data(); long = r3.cohort(d); pre = pd.read_csv(ROOT / 'inputs/robustness_prefixes.csv'); sel = pd.read_csv(ROOT / 'tables/r4_hybrid_selected.csv')
    pre['sigma'] = pre.job_id.str.extract(r'_n(\d+)_').astype(int) / 10000; pre['drop'] = pre.job_id.str.extract(r'_d(\d+)_').astype(int) / 100; pre['rep'] = pre.job_id.str.extract(r'_r(\d+)$').astype(int)
    out = []
    for fold in range(1, 6):
        s = r3.split(fold, d); tr = pd.concat([s['fit'], s['calibration']]); T = H.frame(tr, list(range(8, tr.groupby('cell_id').size().max() - 1, 2)))
        b = int(sel[(sel.fold == fold) & (sel.origin == 20)].config.iloc[0]); models = [XGBRegressor(**H.CONFIGS[b], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[H.FEAT], T.soh_true - T.prior) for sd in SEEDS]
        te = s['test'][s['test'].cell_id.isin(long)]
        for (jid, cid), p in pre[pre.cell_id.isin(te.cell_id.unique())].groupby(['job_id', 'cell_id']):
            g = H.ordered(te[te.cell_id.eq(cid)]).copy(); p = p.sort_values('row'); obs = p.observed.to_numpy(bool)
            g.loc[:19, 'Q'] = p.Q_observed.to_numpy(); keep = np.r_[obs, np.ones(len(g) - 20, bool)]; gg = g[keep].reset_index(drop=True); o = int(obs.sum())
            # truth relative to the preserved original first measurement
            E = pd.DataFrame(H.cell_rows(gg, [o], [c for c in H.STATIC if c != 'age_type']))
            E['soh_true'] = (g.Q.iloc[0] and (te[te.cell_id.eq(cid)].sort_values('timestamp_s').Q.to_numpy()[20:] / g.Q.iloc[0]))
            for sd, m in zip(SEEDS, models):
                out.append(E[['cell_id', 'step', 'soh_true']].assign(pred=np.maximum(E.prior + m.predict(E[H.FEAT]), 1e-6), seed=sd, sigma=p.sigma.iloc[0], drop=p['drop'].iloc[0], rep=p.rep.iloc[0]))
        print('g4 fold', fold, flush=True)
    R = pd.concat(out); R['ape'] = 100 * (R.pred - R.soh_true).abs() / R.soh_true; R.to_csv(ROOT / 'results/r4_g4_robust_pred.csv', index=False)
    hc = R.groupby(['sigma', 'drop', 'rep', 'seed', 'cell_id']).ape.mean().groupby(['sigma', 'drop', 'cell_id']).mean().rename('hybrid')
    rc = pd.read_csv(ROOT / 'inputs/robustness_cell_errors.csv').rename(columns={'noise_sigma_Q0': 'sigma', 'drop_fraction': 'drop'})
    piv = rc.pivot_table(index=['sigma', 'drop', 'cell_id'], columns='method', values='APE').join(hc)
    S = piv.groupby(level=[0, 1]).mean(); S.to_csv(ROOT / 'tables/r4_g4_robust_summary.csv')
    pairs = []
    for (sg, dr), x in piv.groupby(level=[0, 1]):
        dd = (x.hybrid - x.ARIMA_AICc).dropna(); pairs.append(dict(sigma=sg, drop=dr, n=len(dd), hybrid_minus_ARIMA=dd.mean(), p=stats.wilcoxon(dd).pvalue))
    Pp = pd.DataFrame(pairs); Pp['p_holm'] = holm(Pp.p); Pp.to_csv(ROOT / 'tables/r4_g4_robust_pairs.csv', index=False)
    print(S.round(3).to_string()); print(Pp.round(4).to_string(index=False))


# ---------------- G5 Mondrian conformal (Luh, needs r4_uncert fit-only predictions)
def g5():
    P = pd.read_csv(ROOT / 'results/r4_hybrid_fitonly_pred.csv'); P['b'] = np.select([P.step <= 3, P.step <= 6], [P.step, 4], 7); P['res'] = (P.Q_pred - P.Q_true).abs(); rows = []
    for (fold, o, sd), g in P.groupby(['fold', 'origin', 'seed']):
        cal = g[g.role.eq('calibration')]; te = g[g.role.eq('test')]
        for q in (.8, .9):
            rad = {}
            for key, r in cal.groupby(['age_type', 'b']).res:
                r = np.sort(r.to_numpy()); n = len(r); k = int(np.ceil((n + 1) * q)); rad[key] = r[k - 1] if k <= n else np.inf
            h = np.array([rad.get((a, b), np.inf) for a, b in zip(te.age_type, te.b)])
            rows.append(pd.DataFrame(dict(origin=o, nominal=q, age_type=te.age_type.to_numpy(), cell_id=te.cell_id.to_numpy(), cov=(te.res.to_numpy() <= h), w=2 * h / te.Q0.to_numpy())))
    R = pd.concat(rows); c = R.groupby(['origin', 'nominal', 'age_type', 'cell_id']).agg(cov=('cov', 'mean'), w=('w', lambda x: x[np.isfinite(x)].mean() if np.isfinite(x).any() else np.inf), fin=('w', lambda x: np.isfinite(x).mean())).reset_index()
    S = c.groupby(['origin', 'nominal', 'age_type']).agg(coverage=('cov', 'mean'), width_SOH=('w', 'mean'), finite=('fin', 'mean')).reset_index()
    O = c.groupby(['origin', 'nominal']).agg(coverage=('cov', 'mean'), width_SOH=('w', 'mean'), finite=('fin', 'mean')).reset_index()
    S.to_csv(ROOT / 'tables/r4_g5_mondrian_by_regime.csv', index=False); O.to_csv(ROOT / 'tables/r4_g5_mondrian_overall.csv', index=False); print(O.round(3).to_string(index=False)); print(S.round(3).to_string(index=False))


# ---------------- G6 validation-chosen ARIMA weight
def g6():
    W = [0, .25, .5, .75, 1]; out = []
    ref = pd.read_csv(ROOT / 'inputs/r3_reference_predictions.csv'); ar = ref[ref.method.eq('ARIMA_AICc') & ref.role.isin(['validation', 'test'])]
    ar = ar.assign(a=ar.Q_pred / ar.Q0)[['fold', 'origin', 'role', 'seed', 'cell_id', 'row', 'a']]
    h = pd.read_csv(ROOT / 'results/r4_hybrid_pred.csv'); x = h.merge(ar, on=['fold', 'origin', 'role', 'seed', 'cell_id', 'row'], validate='one_to_one'); x['dataset'] = 'luh'
    hs = pd.read_csv(ROOT / 'results/r4_hybrid_snl_pred.csv'); xs = []
    for o in (5, 8):
        c = pd.read_csv(f'{_SNL}/origin{o}/controls_pred.csv'); c = c[c.family.eq('ARIMA_AICc')][['cell_id', 'row', 'Q_pred']].rename(columns={'Q_pred': 'a'})
        xs.append(hs[hs.origin.eq(o)].merge(c, on=['cell_id', 'row'], validate='many_to_one'))
    x = pd.concat([x[['dataset', 'fold', 'origin', 'role', 'seed', 'cell_id', 'row', 'soh_true', 'pred', 'a']], pd.concat(xs).assign(dataset='snl')[['dataset', 'fold', 'origin', 'role', 'seed', 'cell_id', 'row', 'soh_true', 'pred', 'a']]])
    sel = []
    for (dn, o, f), g in x.groupby(['dataset', 'origin', 'fold']):
        v = g[g.role.eq('validation')]; sc = [eq(v.assign(c=w * v.a + (1 - w) * v.pred), 'c').mean() for w in W]; w = W[int(np.argmin(sc))]; sel.append(dict(dataset=dn, origin=o, fold=f, w=w))
        t = g[g.role.eq('test')]; out.append(t.assign(comb=w * t.a + (1 - w) * t.pred, w=w))
    R = pd.concat(out); rows = []
    for (dn, o), g in R.groupby(['dataset', 'origin']):
        e1 = eq(g, 'comb'); e0 = eq(g, 'pred'); ea = eq(g, 'a'); d = e1 - ea; d2 = e1 - e0
        rows.append(dict(dataset=dn, origin=o, weighted=e1.mean(), hybrid=e0.mean(), ARIMA=ea.mean(), diff_vs_ARIMA=d.mean(), p_vs_ARIMA=stats.wilcoxon(d).pvalue, diff_vs_hybrid=d2.mean(), p_vs_hybrid=stats.wilcoxon(d2).pvalue if (d2 != 0).any() else 1.0))
    S = pd.DataFrame(rows); S['p_vs_ARIMA_holm'] = holm(S.p_vs_ARIMA); S.to_csv(ROOT / 'tables/r4_g6_arima_weight.csv', index=False); pd.DataFrame(sel).to_csv(ROOT / 'tables/r4_g6_weights.csv', index=False)
    print(pd.DataFrame(sel).groupby(['dataset', 'origin']).w.apply(list).to_string()); print(S.round(4).to_string(index=False))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
