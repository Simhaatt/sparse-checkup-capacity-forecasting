"""Re-run the fold-winner final fits (same configs/seeds/schedule) to save inner-validation predictions
needed by the validation-chosen selector. Test predictions are re-saved and compared with final_pred.csv."""
import os, sys, json
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import snl_bench as S


def job(j):
    import torch; torch.set_num_threads(1)
    import exp_lib as X; X.COMMON_START = S.ORIGIN
    d = S.data(); s = S.split(j['fold'], d); fam, cfg, seed = j['family'], j['cfg'], j['seed']
    if fam in ('compact_lstm', 'gru', 'large_lstm'):
        _, pred, (m, sx, sy) = X.train_eval(s['fit'], s['validation'], s['test'], 'gru' if fam == 'gru' else 'lstm', cfg, S.FEATS, seed, 'final')
        yt, yp, meta, _ = X.rollout_predict(m, sx, sy, s['validation'], S.FEATS, cfg['L'])
        v = meta.copy(); v['Q_true'] = yt[:, 0]; v['Q_pred'] = yp[:, 0]
    else:
        xx, yy = X.build_sequences(s['fit'], S.FEATS, cfg['L'])
        m = S.tree_model(fam, cfg, seed); m.fit(xx.reshape(len(xx), -1), yy.ravel())
        v = S.predict_tree(m, s['validation'], cfg['L']); pred = S.predict_tree(m, s['test'], cfg['L'])
    out = []
    for frame, role in ((pred, 'test'), (v, 'validation')):
        f = frame[['cell_id', 'row', 'step', 'age_type', 'Q_true', 'Q_pred']].copy()
        f['family'] = fam; f['fold'] = j['fold']; f['seed'] = seed; f['role'] = role; out.append(f)
    return pd.concat(out)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    w = pd.read_csv(S.OUT / 'fold_selected_settings.csv')
    jobs = [dict(family=r.family, fold=int(r.fold), cfg=json.loads(r.cfg_json), seed=sd) for r in w.itertuples() for sd in S.SEEDS]
    with ProcessPoolExecutor(2, max_tasks_per_child=10) as ex:
        res = pd.concat(list(ex.map(job, jobs)), ignore_index=True)
    res.to_csv(S.OUT / 'final_pred_with_validation.csv', index=False)
    old = pd.read_csv(S.OUT / 'final_pred.csv')
    t = res[res.role.eq('test')].merge(old, on=['cell_id', 'row', 'family', 'fold', 'seed'], suffixes=('', '_old'))
    print(f'origin {S.ORIGIN}: test rows matched {len(t)}/{len(old)}; max |dQ_pred| vs saved finals = {(t.Q_pred - t.Q_pred_old).abs().max():.3e}')
