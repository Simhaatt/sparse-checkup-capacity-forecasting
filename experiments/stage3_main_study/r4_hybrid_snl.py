"""Round 4 D on Sandia: same hybrid as r4_hybrid.py; Sandia folds/partitions from the Sandia benchmark; origins 5 and 8.
Prior axis: planned EFC (k_exp; index-free), lag-one, slope<=0. Training on 'fit' cells, selection on 'validation'."""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, os
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, _SNL); sys.path.insert(0, str(Path(__file__).resolve().parent))
import snl_bench as S
import r4_hybrid as H
from xgboost import XGBRegressor
OUT = Path(__file__).resolve().parent
STATIC = ['temperature', 'c_rate_chg', 'c_rate_dischg', 'soc_window']
H.STATIC = STATIC + ['age_type']; FEAT = ['step', 'origin', 'prior', 'last_soh', 'last_res', 'mean_abs_res', 'slope', 'gap', 'k_prev', 'Q0'] + STATIC + ['age_type']
H.ordered = lambda g: g.sort_values('timestamp_s', kind='mergesort').reset_index(drop=True)


def frame(d, origins):
    return pd.DataFrame([r for _, g in d.groupby('cell_id') for r in H.cell_rows(g, origins, STATIC)])


def main():
    d = S.data(); d['Q'] = d.Q * d.Q0  # back to Ah so SOH = Q/Q_first inside cell_rows
    out, sel = [], []
    for o in (5, 8):
        part = pd.read_csv(f'{_SNL}/origin{o}/partitions.csv'); n = d.groupby('cell_id').size(); sc = set(n[n > o].index)
        for fold in range(1, 6):
            roles = part[part.fold.eq(fold)].set_index('cell_id').role
            tr = d[d.cell_id.map(roles).eq('fit')]; va = d[d.cell_id.map(roles).eq('validation') & d.cell_id.isin(sc)]; te = d[d.cell_id.map(roles).eq('test') & d.cell_id.isin(sc)]
            T = frame(tr, list(range(3, tr.groupby('cell_id').size().max() - 1))); V = frame(va, [o]); E = frame(te, [o])
            scores = []
            for c in H.CONFIGS:
                m = XGBRegressor(**c, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=42).fit(T[FEAT], T.soh_true - T.prior)
                V['pred'] = np.maximum(V.prior + m.predict(V[FEAT]), 1e-6); scores.append(H.ecm(V))
            best = int(np.argmin(scores)); sel.append(dict(origin=o, fold=fold, config=best, val_MAPE=scores[best], train_rows=len(T)))
            for sd in (42, 52, 62):
                m = XGBRegressor(**H.CONFIGS[best], subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=sd).fit(T[FEAT], T.soh_true - T.prior)
                for role, F in (('validation', V), ('test', E)):
                    f = F[['cell_id', 'row', 'step', 'age_type', 'soh_true', 'prior']].copy(); f['pred'] = np.maximum(F.prior + m.predict(F[FEAT]), 1e-6)
                    f['origin'] = o; f['fold'] = fold; f['seed'] = sd; f['role'] = role; out.append(f)
        p = pd.concat([x for x in out if x.origin.iloc[0] == o]); t = p[p.role.eq('test')]
        ape = 100 * (t.pred - t.soh_true).abs() / t.soh_true
        print('SANDIA origin', o, 'hybrid', round(ape.groupby([t.seed, t.cell_id]).mean().groupby(level=1).mean().mean(), 3),
              'prior-only', round((100 * (t.prior - t.soh_true).abs() / t.soh_true).groupby(t.cell_id).mean().mean(), 3), 'cells', t.cell_id.nunique(), flush=True)
    pd.concat(out).to_csv(OUT / 'results/r4_hybrid_snl_pred.csv', index=False); pd.DataFrame(sel).to_csv(OUT / 'tables/r4_hybrid_snl_selected.csv', index=False)


if __name__ == '__main__':
    main()
