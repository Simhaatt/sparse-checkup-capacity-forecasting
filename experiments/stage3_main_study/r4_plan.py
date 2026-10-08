"""Round 4 pre-declaration (written before any round-4 model is fitted). Creates PLAN_R4.json.

Motivation (from round-3 results, stated openly): (i) structures were selected only at origin 20 and replayed at
10/15; (ii) final models were fitted on the 'fit' role only (validation and 20 calibration cells withheld) while the
learning curve shows strong data dependence; (iii) capacity-history-only inputs beat full descriptors for neural
models in the round-3 ablation. Round 4 tests protocol fixes and two additions; every choice uses inner validation
only; outer-test results of every arm are reported whatever their sign.
"""
import json, numpy as np, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FAM = ['compact_lstm', 'gru', 'large_lstm', 'xgb', 'rf']
LS = {10: [4, 6, 8, 10], 15: [6, 9, 12, 15], 20: [8, 12, 16, 20]}
def space(f, o):
    base = dict(input=['full', 'history_only'], L=LS[o])
    if f == 'compact_lstm': base.update(hidden=[16, 32, 48, 64], layers=[1], H=[1, 3, 5, 8])
    if f == 'gru': base.update(hidden=[32, 64, 96, 128], layers=[1, 2], H=[1, 3, 5, 8])
    if f == 'large_lstm': base.update(hidden=[128, 160, 192, 256], layers=[1, 2], H=[1, 3, 5, 8])
    if f == 'xgb': base.update(n_estimators=[100, 200, 300], depth=[2, 3, 4, 6], subsample=[.7, .85, 1.], colsample=[.7, .85, 1.])
    if f == 'rf': base.update(n_estimators=[100, 200, 300], depth=[4, 8, 12, None], leaf=[1, 2, 4, 8], max_features=[.5, .75, 1.])
    return base
trials = []
for o in (10, 15, 20):
    for fi, f in enumerate(FAM):
        r = np.random.default_rng(20261007 + 1000 * o + 100 * fi); seen = set(); sp = space(f, o)
        while len(seen) < 20:
            cfg = {k: v[r.integers(len(v))] for k, v in sp.items()}
            cfg = {k: (v.item() if hasattr(v, 'item') else v) for k, v in cfg.items()}
            if f in FAM[:3]:
                if cfg['L'] + cfg['H'] > 29: continue
                cfg['lr'] = float(np.exp(r.uniform(np.log(3e-4), np.log(3e-3))))
            elif f == 'xgb':
                cfg['lr'] = float(np.exp(r.uniform(np.log(.02), np.log(.2))))
            key = json.dumps(cfg, sort_keys=True)
            if key in seen: continue
            seen.add(key); trials.append(dict(origin=o, family=f, trial=len(seen), cfg=cfg))
plan = dict(
    created='2026-10-07, before any round-4 fit',
    data='Same Luh-Blank 104 scored cells, same round-3 partitions (audit/cell_partitions.csv), same round-3 trainer (trainer/exp_lib.py) and r3.py helpers, copied unmodified.',
    R4A_origin_specific_search='For each origin in {10,15,20} and each family: 20 random settings (equal budget) drawn from an origin-specific space in which the information set (full F7+Q or capacity-history-only) is a tuned dimension. Fit on role=fit, select on equal-cell recursive validation Q-MAPE (role=validation), seed 42, search schedule.',
    R4B_refit='Winner per family/fold/origin refitted with seeds 42/52/62 (final schedule) on fit+calibration cells; neural early stopping on the validation cells; trees on fit+calibration+validation. Calibration cells therefore no longer support split-conformal claims for R4 models (round-3 intervals remain the conformal evidence).',
    R4C_pooled='History-only SOH (Q/Q0) GRU and XGBoost. Arm pooled: training adds every cell of Sandia(SNL, 80), Oxford(8) and Kirkaldy(40) to the Luh-Blank fit+calibration cells; arm source_only: Luh-Blank fit+calibration only; same configuration (the fold/origin R4A validation-best history-only trial of that family), same seeds, same validation/test cells. Mirror study on Sandia (origins 5, 8; Sandia folds; configuration = Sandia fold-selected setting with history-only input).',
    R4D_hybrid='Per-cell prior = lag-one sqrt fit on k (days when k is constant in the prefix), slope<=0, as round-3 sqrt_k_lag1. A direct XGBoost learns the residual SOH - prior from training cells over simulated origins (every 2nd checkup from 8, or from 3 for Sandia); features: steps ahead, prior forecast, last SOH, last prior residual, mean residual over prefix, recent slope, origin length, coordinate gap, static descriptors. 6 pre-listed XGB settings, chosen on validation cells at the evaluation origin; seeds 42/52/62.',
    combinations='Equal-weight, per seed: ARIMA + R4 XGB; ARIMA + R4 best-validation neural; ARIMA + R4D hybrid.',
    statistics='Equal-cell Q-MAPE (steps -> seeds -> cells); bootstrap 10000 seed 20261005; paired Wilcoxon on 104 cells; Holm within named families r4_vs_ARIMA_origin{o} and r4_vs_round3_origin{o}. No equivalence claims.',
    trials=trials)
(ROOT / 'PLAN_R4.json').write_text(json.dumps(plan, indent=1, default=str))
print('trials', len(trials), 'sha256', hashlib.sha256((ROOT / 'PLAN_R4.json').read_bytes()).hexdigest())
