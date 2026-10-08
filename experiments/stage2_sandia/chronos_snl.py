"""Chronos-Bolt-small zero-shot on each scored Sandia cell's measured SOH prefix (same prefix as all methods).
Weights: amazon/chronos-bolt-small revision 772f3d25d38aec6d914c8949dab4462e2d46f5d8 (copied from the laptop's pinned,
sha256-verified download used in round 3). Checkup spacing is irregular; Chronos sees only the ordered values."""
import os, sys, hashlib, numpy as np, pandas as pd, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import snl_bench as S
W = Path(os.environ.get('CHRONOS_DIR', HERE / 'chronos-bolt-small'))
assert hashlib.sha256((W / 'model.safetensors').read_bytes()).hexdigest() == '06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd'
from chronos import BaseChronosPipeline
torch.set_num_threads(1)
pipe = BaseChronosPipeline.from_pretrained(str(W), device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
qs = list(pipe.quantiles); d = S.data(); sc = S.scoreable(d); rows = []
for cid, g0 in d[d.cell_id.isin(sc)].groupby('cell_id'):
    g = g0.sort_values('k_exp', kind='mergesort').reset_index(drop=True); y = g.Q.to_numpy(); h = len(g) - S.ORIGIN
    with torch.no_grad():
        f = pipe.predict(torch.tensor(y[:S.ORIGIN], dtype=torch.float32), prediction_length=h).cpu().numpy()[0]
    rows.append(pd.DataFrame(dict(cell_id=cid, row=np.arange(S.ORIGIN, len(g)), step=np.arange(1, h + 1), age_type=g.age_type.iloc[S.ORIGIN:].to_numpy(),
                                  Q_true=y[S.ORIGIN:], Q_pred=np.maximum(f[qs.index(.5), :h], 1e-6), q10=f[qs.index(.1), :h], q90=f[qs.index(.9), :h],
                                  family='ChronosBolt_small_zero', seed=0)))
pd.concat(rows, ignore_index=True).to_csv(S.OUT / 'chronos_pred.csv', index=False); print('chronos done', len(rows), 'cells')
