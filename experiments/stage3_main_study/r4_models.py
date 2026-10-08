"""Round 4 A/B: origin-specific equal-budget search (input set tuned) and refit of winners on fit+calibration.
Usage: python r4_models.py search|final [--workers 2]"""
from r3 import *
import copy, pickle, argparse
from concurrent.futures import ProcessPoolExecutor
PLAN4 = json.loads((ROOT / 'PLAN_R4.json').read_text())
NEURAL = ['compact_lstm', 'gru', 'large_lstm']


def feats(inp):
    return F7 if inp == 'full' else []


def fit_job(job):
    import torch, exp_lib as X
    torch.set_num_threads(1); X.COMMON_START = job['origin']
    p = split(job['fold']); cfg = copy.deepcopy(job['cfg']); inp = cfg.pop('input'); fs = feats(inp)
    fam = job['family']; t0 = time.perf_counter(); c0 = time.process_time()
    va = p['validation']
    if job['stage'] == 'search':
        tr = p['fit']
    else:
        tr = pd.concat([p['fit'], p['calibration']] + ([] if fam in NEURAL else [p['validation']]), ignore_index=True)
    if fam in NEURAL:
        _, _, (m, sx, sy) = X.train_eval(tr, va, None, 'gru' if fam == 'gru' else 'lstm', cfg, fs, job['seed'], job['schedule'])
        pv = predict_neural(X, m, sx, sy, va, fs, cfg['L'])
        pt = predict_neural(X, m, sx, sy, p['test'], fs, cfg['L']) if job['stage'] == 'final' else None
        size = X.serialized_bytes(m); npar = X.count_params(m)
    else:
        xx, yy = X.build_sequences(tr, fs, cfg['L']); m = tree_model(fam, cfg, job['seed']); m.fit(xx.reshape(len(xx), -1), yy.ravel())
        pv = predict_tree(m, va, fs, cfg['L'], job['origin'])
        pt = predict_tree(m, p['test'], fs, cfg['L'], job['origin']) if job['stage'] == 'final' else None
        size = len(pickle.dumps(m)); npar = np.nan
    row = {k: v for k, v in job.items() if k != 'cfg'}
    row.update(cfg_json=json.dumps(job['cfg'], sort_keys=True), input=inp, val_Q_MAPE_equal_cell=equal_cell(pv), train_cells=tr.cell_id.nunique(),
               fit_wall_seconds=time.perf_counter() - t0, fit_cpu_seconds=time.process_time() - c0, serialized_bytes=size, parameters=npar)
    out = []
    for frame, role in ((pv, 'validation'), (pt, 'test')):
        if frame is None: continue
        f = frame[['cell_id', 'row', 'step', 'age_type', 'Q_true', 'Q_pred']].copy()
        for k in ('job_id', 'family', 'fold', 'seed', 'origin', 'stage'): f[k] = job[k]
        f['role'] = role; out.append(f)
    return row, pd.concat(out, ignore_index=True)


def run(jobs, name, workers):
    rp, pp = ROOT / 'results' / f'r4_{name}_runs.csv', ROOT / 'results' / f'r4_{name}_pred.csv'
    done = set(pd.read_csv(rp).job_id) if rp.exists() else set(); todo = [j for j in jobs if j['job_id'] not in done]
    print('STAGE', name, 'done', len(done), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, max_tasks_per_child=4) as ex:
        for k, (row, pred) in enumerate(ex.map(fit_job, todo, chunksize=1)):
            append(pp, pred); append(rp, pd.DataFrame([row]))
            print('DONE', name, len(done) + k + 1, '/', len(jobs), row['job_id'], round(row['val_Q_MAPE_equal_cell'], 4), round(row['fit_wall_seconds'], 1), flush=True)


def search_jobs():
    J = []
    for o in (10, 15, 20):
        for f in range(1, 6):
            for t in PLAN4['trials']:
                if t['origin'] == o:
                    J.append(dict(job_id=f"s_o{o}_f{f}_{t['family']}_t{t['trial']:02d}", stage='search', family=t['family'], fold=f, trial=t['trial'], cfg=t['cfg'], seed=42, origin=o, schedule='search'))
    # cheap trees first within each origin so early winners exist; neural interleaved
    return J


def winners(origins=(10, 15, 20)):
    r = pd.read_csv(ROOT / 'results/r4_search_runs.csv'); r = r[r.origin.isin(origins)]
    assert (r.groupby(['origin', 'fold', 'family']).size() == 20).all() and len(r) == 500 * len(origins), 'search incomplete'
    w = r.sort_values(['val_Q_MAPE_equal_cell', 'trial'], kind='mergesort').groupby(['origin', 'fold', 'family']).head(1)
    w.to_csv(ROOT / f"tables/r4_fold_selected_settings_o{'_'.join(map(str, origins))}.csv", index=False); return w


def final_jobs(origins=(10, 15, 20)):
    return [dict(job_id=f"f_o{r.origin}_f{r.fold}_{r.family}_s{sd}", stage='final', family=r.family, fold=int(r.fold), trial=int(r.trial), cfg=json.loads(r.cfg_json),
                 seed=sd, origin=int(r.origin), schedule='final') for r in winners(origins).itertuples() for sd in SEEDS]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--origin', type=int, default=0); a = ap.parse_args()
    org = (a.origin,) if a.origin else (10, 15, 20)
    run([j for j in search_jobs() if j['origin'] in org] if a.stage == 'search' else final_jobs(org), a.stage, a.workers)
    print('R4_STAGE_COMPLETE', a.stage, flush=True)
