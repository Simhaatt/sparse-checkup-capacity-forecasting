"""R4F: fine-tuned Chronos-Bolt-small on sparse SOH checkups (per outer fold), Luh (10/15/20) and Sandia (5/8).
Training cells only; checkpoint (200/400/600 steps) chosen by inner-validation equal-cell MAPE averaged over origins."""
import os as _os
from pathlib import Path as _Path
_SNL = _os.environ.get('SNL_BENCH', str(_Path(__file__).resolve().parents[1] / 'stage2_sandia'))  # repo edit: was '/home/claude/sandia/bench'
import sys, copy, numpy as np, pandas as pd, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, _SNL)
import r3
from chronos import BaseChronosPipeline
W = '/mnt/user-data/uploads/pinn/JES_capacity_retuning/external/chronos-bolt-small'
import os
CTX, BATCH = 32, 32
FULL = os.environ.get('CFT_FULL') == '1'; LR = float(os.environ.get('CFT_LR', '5e-5')); STEPS = int(os.environ.get('CFT_STEPS', '600'))
CKPT = tuple(int(x) for x in os.environ.get('CFT_CKPT', '200,400,600').split(',')); TAG = os.environ.get('CFT_TAG', '')
torch.set_num_threads(1)


def seqs(d):
    return {cid: (g.sort_values('timestamp_s').Q / g.sort_values('timestamp_s').Q.iloc[0]).to_numpy(np.float32) for cid, g in d.groupby('cell_id')}


def roles(name, fold, o=None):
    if name == 'luh':
        d = r3.data(); s = r3.split(fold, d); long = r3.cohort(d)
        return pd.concat([s['fit'], s['calibration']]), s['validation'][s['validation'].cell_id.isin(long)], s['test'][s['test'].cell_id.isin(long)]
    import snl_bench as S
    d = S.data().copy(); p = pd.read_csv(f'{_SNL}/origin{o}/partitions.csv'); p = p[p.fold.eq(fold)].set_index('cell_id').role; r = d.cell_id.map(p)
    n = d.groupby('cell_id').size(); sc = set(n[n > o].index)
    return d[r.eq('fit')], d[r.eq('validation') & d.cell_id.isin(sc)], d[r.eq('test') & d.cell_id.isin(sc)]


def predict(model, S, o):
    rows = []
    with torch.no_grad():
        for cid, y in S.items():
            if len(y) <= o: continue
            ctx = torch.full((1, CTX), float('nan')); c = y[:o][-CTX:]; ctx[0, -len(c):] = torch.tensor(c)
            q = model(ctx).quantile_preds[0].numpy(); med = q[q.shape[0] // 2, :len(y) - o]
            rows.append(pd.DataFrame(dict(cell_id=cid, row=np.arange(o, len(y)), step=np.arange(1, len(y) - o + 1), soh_true=y[o:], pred=np.maximum(med, 1e-6),
                                          q10=q[0, :len(y) - o], q90=q[-1, :len(y) - o])))
    return pd.concat(rows)


def ecm(p): return float((100 * (p.pred - p.soh_true).abs() / p.soh_true).groupby(p.cell_id).mean().mean())


def run(name, origins):
    out, sel = [], []
    for fold in range(1, 6):
        tr, va, te = roles(name, fold, origins[0]) if name == 'snl' else roles(name, fold)
        if name == 'snl':
            vas = {o: roles(name, fold, o)[1] for o in origins}; tes = {o: roles(name, fold, o)[2] for o in origins}
        else:
            vas = {o: va for o in origins}; tes = {o: te for o in origins}
        pipe = BaseChronosPipeline.from_pretrained(W, device_map='cpu', torch_dtype=torch.float32, local_files_only=True); model = pipe.model
        for n_, p_ in model.named_parameters(): p_.requires_grad = FULL or not n_.startswith(('encoder', 'shared', 'input_patch'))  # compute amendment: encoder frozen
        H = model.chronos_config.prediction_length; Str = seqs(tr); g = torch.Generator().manual_seed(42); rng = np.random.default_rng(42)
        samples = [(cid, c) for cid, y in Str.items() for c in range(3, len(y))]
        opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=0.0); best = (np.inf, None, 0)
        for step in range(1, STEPS + 1):
            model.train(); idx = rng.integers(len(samples), size=BATCH); ctx = torch.full((BATCH, CTX), float('nan')); tgt = torch.full((BATCH, H), float('nan'))
            for b, k in enumerate(idx):
                cid, c = samples[k]; y = Str[cid]; cc = y[:c][-CTX:]; ctx[b, -len(cc):] = torch.tensor(cc); tt = y[c:c + H]; tgt[b, :len(tt)] = torch.tensor(tt)
            loss = model(ctx, target=tgt).loss; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step in CKPT:
                model.eval(); v = np.mean([ecm(predict(model, seqs(vas[o]), o)) for o in origins])
                print(name, 'fold', fold, 'step', step, 'loss', round(float(loss), 4), 'val', round(v, 3), flush=True)
                if v < best[0]: best = (v, copy.deepcopy(model.state_dict()), step)
        model.load_state_dict(best[1]); model.eval(); sel.append(dict(dataset=name, fold=fold, step=best[2], val=best[0]))
        for o in origins:
            p = predict(model, seqs(tes[o]), o); p['origin'] = o; p['fold'] = fold; p['method'] = 'ChronosBolt_small_finetuned' + TAG; out.append(p)
    P = pd.concat(out); P.to_csv(ROOT / f'results/r4_chronos_ft_{name}{TAG}_pred.csv', index=False); pd.DataFrame(sel).to_csv(ROOT / f'tables/r4_chronos_ft_{name}{TAG}_selected.csv', index=False)
    for o in origins:
        x = P[P.origin.eq(o)]; print(name, 'origin', o, 'fine-tuned Chronos equal-cell MAPE', round(ecm(x), 3), 'q10-q90 coverage', round(((x.soh_true >= x.q10) & (x.soh_true <= x.q90)).groupby(x.cell_id).mean().mean(), 3), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'snl': run('snl', (5, 8))
    else: run('luh', (10, 15, 20))
