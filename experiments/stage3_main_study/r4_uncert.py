"""R4F intervals + decisions for the hybrid.
Luh: V1 hybrid refitted on 'fit' cells only (calibration withheld), selection on validation; split-conformal per
fold/origin/seed/step on the 20 calibration cells (rank ceil((n+1)q), infinite if > n); coverage/width on test.
Decisions (common script): 80%-SOH crossing P/R/F1, last-step APE, signed bias, optimistic share.
Sandia: CV+-style intervals and decisions via the Sandia extras functions."""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, _SNL)
import r3, r4_hybrid as H
from xgboost import XGBRegressor


def luh_intervals():
    d = r3.data(); long = r3.cohort(d); out = []
    for fold in range(1, 6):
        s = r3.split(fold, d); tr = s['fit']; T = H.frame(tr, list(range(8, tr.groupby('cell_id').size().max() - 1, 2)))
        for o in (10, 15, 20):
            V = H.frame(s['validation'][s['validation'].cell_id.isin(long)], [o]); C = H.frame(s['calibration'][s['calibration'].cell_id.isin(long)], [o]); E = H.frame(s['test'][s['test'].cell_id.isin(long)], [o])
            sc = []
            for c in H.CONFIGS:
                m = XGBRegressor(**c, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=42).fit(T[H.FEAT], T.soh_true - T.prior)
                V['pred'] = np.maximum(V.prior + m.predict(V[H.FEAT]), 1e-6); sc.append(H.ecm(V))
            b = int(np.argmin(sc))
            for sd in r3.SEEDS:
                m = XGBRegressor(**H.CONFIGS[b], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[H.FEAT], T.soh_true - T.prior)
                for role, F in (('calibration', C), ('test', E)):
                    f = F[['cell_id', 'row', 'step', 'age_type', 'soh_true', 'Q0']].copy(); f['pred'] = np.maximum(F.prior + m.predict(F[H.FEAT]), 1e-6)
                    f['Q_true'] = f.soh_true * f.Q0; f['Q_pred'] = f.pred * f.Q0; f['role'] = role; f['fold'] = fold; f['origin'] = o; f['seed'] = sd; out.append(f)
        print('intervals fold', fold, flush=True)
    P = pd.concat(out, ignore_index=True); P.to_csv(ROOT / 'results/r4_hybrid_fitonly_pred.csv', index=False)
    rows = []
    for (fold, o, sd), g in P.groupby(['fold', 'origin', 'seed']):
        cal = g[g.role.eq('calibration')]; te = g[g.role.eq('test')].copy(); te['res'] = (te.Q_pred - te.Q_true).abs()
        for q in (.8, .9):
            rad = {}
            for st, r in cal.assign(res=(cal.Q_pred - cal.Q_true).abs()).groupby('step').res:
                r = np.sort(r.to_numpy()); n = len(r); k = int(np.ceil((n + 1) * q)); rad[st] = r[k - 1] if k <= n else np.inf
            h = te.step.map(rad).fillna(np.inf)
            rows.append(pd.DataFrame(dict(fold=fold, origin=o, seed=sd, nominal=q, cell_id=te.cell_id, covered=(te.res <= h).to_numpy(), width_Ah=2 * h.to_numpy(), width_SOH=2 * h.to_numpy() / te.Q0.to_numpy())))
    R = pd.concat(rows); R['finite'] = np.isfinite(R.width_Ah)
    c = R.groupby(['origin', 'nominal', 'cell_id']).agg(cov=('covered', 'mean'), wA=('width_Ah', lambda x: x[np.isfinite(x)].mean()), wS=('width_SOH', lambda x: x[np.isfinite(x)].mean()), fin=('finite', 'mean')).reset_index()
    S = c.groupby(['origin', 'nominal']).agg(coverage_equal_cell=('cov', 'mean'), mean_width_Ah=('wA', 'mean'), mean_width_SOH=('wS', 'mean'), finite_fraction=('fin', 'mean')).reset_index()
    t = P[P.role.eq('test')]; ape = 100 * (t.Q_pred - t.Q_true).abs() / t.Q_true
    acc = ape.groupby([t.origin, t.seed, t.cell_id]).mean().groupby(level=[0, 2]).mean().groupby(level=0).mean()
    S['fitonly_point_MAPE'] = S.origin.map(acc); S.to_csv(ROOT / 'tables/r4_hybrid_conformal_luh.csv', index=False); print(S.round(4).to_string(index=False))


def decisions(df):
    """df: method, origin, seed, cell_id, step, soh_true, soh_pred."""
    out = []
    for (m, o), x in df.groupby(['method', 'origin']):
        per = []
        for (sd, cid), y in x.sort_values('step').groupby(['seed', 'cell_id']):
            tc = y.step[y.soh_true <= .8]; pc = y.step[y.soh_pred <= .8]; last = y.iloc[-1]
            per.append(dict(seed=sd, cell_id=cid, tcross=len(tc) > 0, pcross=len(pc) > 0, err=(pc.iloc[0] - tc.iloc[0]) if len(tc) and len(pc) else np.nan,
                            lape=100 * abs(last.soh_pred - last.soh_true) / last.soh_true, sgn=100 * (last.soh_pred - last.soh_true) / last.soh_true))
        p = pd.DataFrame(per); f1 = []; pr = []; rc = []
        for sd, z in p.groupby('seed'):
            tp = (z.tcross & z.pcross).sum(); fp = (~z.tcross & z.pcross).sum(); fn = (z.tcross & ~z.pcross).sum()
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan); pr.append(tp / (tp + fp) if tp + fp else np.nan); rc.append(tp / (tp + fn) if tp + fn else np.nan)
        cell = p.groupby('cell_id')[['lape', 'sgn']].mean()
        out.append(dict(method=m, origin=o, true_events=int(p[p.seed == p.seed.iloc[0]].tcross.sum()), precision=np.nanmean(pr), recall=np.nanmean(rc), F1=np.nanmean(f1),
                        last_APE=cell.lape.mean(), signed_bias_pct=cell.sgn.mean(), optimistic_share=(cell.sgn > 0).mean(), abs_cross_step_err=p.err.abs().mean()))
    return pd.DataFrame(out)


def luh_decisions():
    ref = pd.read_csv(ROOT / 'inputs/r3_reference_predictions.csv'); ref = ref[ref.role.eq('test')]
    ref['soh_true'] = ref.Q_true / ref.Q0; ref['soh_pred'] = ref.Q_pred / ref.Q0
    h = pd.read_csv(ROOT / 'results/r4_hybrid_pred.csv'); h = h[h.role.eq('test')].copy(); h['soh_pred'] = h.pred; h['method'] = 'hybrid_sqrt_xgb'
    ar = ref[ref.method.eq('ARIMA_AICc')][['origin', 'seed', 'cell_id', 'row', 'soh_pred']]
    a2 = h.merge(ar, on=['origin', 'seed', 'cell_id', 'row'], suffixes=('', '_a')); a2['soh_pred'] = (a2.soh_pred + a2.soh_pred_a) / 2; a2['method'] = 'avg_ARIMA_hybrid'
    cols = ['method', 'origin', 'seed', 'cell_id', 'step', 'soh_true', 'soh_pred']
    D = decisions(pd.concat([ref[cols], h[cols], a2[cols]], ignore_index=True)); D.to_csv(ROOT / 'tables/r4_decisions_luh.csv', index=False)
    print(D.round(3).sort_values(['origin', 'last_APE']).to_string(index=False))


def snl_extras():
    import os, importlib
    out_c, out_d = [], []
    h = pd.read_csv(ROOT / 'results/r4_hybrid_snl_pred.csv')
    for o in (5, 8):
        os.environ['SNL_ORIGIN'] = str(o)
        import snl_bench as S, extras as E
        importlib.reload(S); importlib.reload(E)
        x = h[h.origin.eq(o)].rename(columns={'soh_true': 'Q_true', 'pred': 'Q_pred'}).copy(); x['family'] = 'hybrid_sqrt_xgb'
        t = x[x.role.eq('test')].copy(); t['ape'] = 100 * (t.Q_pred - t.Q_true).abs() / t.Q_true
        c = E.conformal(t); c['origin'] = o; out_c.append(c); d = E.decisions(t); d['origin'] = o; out_d.append(d)
    C = pd.concat(out_c); Dd = pd.concat(out_d); C.to_csv(ROOT / 'tables/r4_hybrid_conformal_snl.csv', index=False); Dd.to_csv(ROOT / 'tables/r4_decisions_snl_hybrid.csv', index=False)
    print(C.round(3).to_string(index=False)); print(Dd.round(3).to_string(index=False))


if __name__ == '__main__':
    luh_decisions(); snl_extras(); luh_intervals()
