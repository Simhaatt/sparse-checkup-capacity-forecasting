"""Round-4 decision metrics with the ROUND-3 rule (fixes r4_uncert.decisions, which had no eligibility filter):
threshold 0.8*Q0; eligible if last prefix capacity >= threshold; crossing = first future checkup strictly below;
no crossing censored at H+1; P/R/F1 per seed then averaged; last-step APE; signed % bias and optimistic share over rows (equal-cell)."""
import numpy as np, pandas as pd, r3
from pathlib import Path
ROOT = Path(__file__).resolve().parent
d = r3.data().sort_values(['cell_id', 'timestamp_s']); d['r'] = d.groupby('cell_id').cumcount(); d['soh'] = d.Q / d.Q0
prefix_last = {o: d[d.r.eq(o - 1)].set_index('cell_id').soh for o in (10, 15, 20)}

def load():
    ref = pd.read_csv(ROOT / 'inputs/r3_reference_predictions.csv'); ref = ref[ref.role.eq('test')]
    ref = ref.assign(soh_true=ref.Q_true / ref.Q0, soh_pred=ref.Q_pred / ref.Q0)
    h = pd.read_csv(ROOT / 'results/r4_hybrid_pred.csv'); h = h[h.role.eq('test')].assign(soh_pred=lambda x: x.pred, method='hybrid_sqrt_xgb')
    ar = ref[ref.method.eq('ARIMA_AICc')][['origin', 'cell_id', 'row', 'soh_pred']].drop_duplicates(['origin', 'cell_id', 'row'])
    a2 = h.merge(ar, on=['origin', 'cell_id', 'row'], suffixes=('', '_a')); a2['soh_pred'] = (a2.soh_pred + a2.soh_pred_a) / 2; a2['method'] = 'avg_ARIMA_hybrid'
    cols = ['method', 'origin', 'seed', 'cell_id', 'row', 'soh_true', 'soh_pred']
    return pd.concat([ref[cols], h[cols], a2[cols]], ignore_index=True)

def metrics(x, o):
    out = []
    for sd, z in x.groupby('seed'):
        rec = []
        for cid, y in z.sort_values('row').groupby('cell_id'):
            H = len(y); el = prefix_last[o][cid] >= 0.8
            t = np.where(y.soh_true.to_numpy() < 0.8)[0]; p = np.where(y.soh_pred.to_numpy() < 0.8)[0]
            ts = t[0] + 1 if len(t) else H + 1; ps = p[0] + 1 if len(p) else H + 1
            ape = 100 * (y.soh_pred - y.soh_true) / y.soh_true
            rec.append(dict(cell_id=cid, el=el, tc=len(t) > 0, pc=len(p) > 0, ts=ts, ps=ps, last=abs(ape.iloc[-1]), sgn=ape.mean(), opt=(ape > 0).mean()))
        r = pd.DataFrame(rec); e = r[r.el]
        tp = (e.tc & e.pc).sum(); fp = (~e.tc & e.pc).sum(); fn = (e.tc & ~e.pc).sum()
        both = e[e.tc & e.pc]
        out.append(dict(seed=sd, eligible=len(e), true_events=int(e.tc.sum()), tp=tp, precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan,
                        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, both_cross_error=(both.ps - both.ts).abs().mean(), censored_cross_error=(e.ps - e.ts).abs().mean(),
                        last=r.set_index('cell_id')['last'], sgn=r.set_index('cell_id').sgn, opt=r.set_index('cell_id').opt))
    o_ = pd.DataFrame(out)
    cellavg = lambda k: pd.concat(list(o_[k]), axis=1).mean(axis=1).mean()
    return dict(eligible=o_.eligible.iloc[0], true_events=o_.true_events.iloc[0], tp=o_.tp.mean(), precision=o_.precision.mean(), recall=o_.recall.mean(), F1=o_.F1.mean(),
                last_APE=cellavg('last'), signed_bias_pct=cellavg('sgn'), optimistic_share=cellavg('opt'), both_cross_error=o_.both_cross_error.mean(), censored_cross_error=o_.censored_cross_error.mean())

if __name__ == '__main__':
    a = load(); rows = []
    for (m, o), x in a.groupby(['method', 'origin']):
        rows.append(dict(method=m, origin=o, **metrics(x, o)))
    D = pd.DataFrame(rows); D.to_csv(ROOT / 'tables/r4_decisions_luh_r3rule.csv', index=False)
    print(D[D.method.isin(['ARIMA_AICc', 'avg_ARIMA_XGB', 'avg_ARIMA_XGB_NN', 'regime_origin_selector', 'rf', 'xgb', 'hybrid_sqrt_xgb', 'avg_ARIMA_hybrid'])].round(4).sort_values(['origin', 'last_APE']).to_string(index=False))
