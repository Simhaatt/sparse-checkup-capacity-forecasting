"""Assemble REPORT_R4.md from round-4 tables (tolerant to missing pieces; missing items are listed as pending)."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parent; T = ROOT / 'tables'; R = ROOT / 'results'
L = []; pending = []


def tab(path, title, fmt=4, index=False):
    p = T / path
    if not p.exists(): pending.append(title); return None
    df = pd.read_csv(p); L.append(f'\n### {title}\n'); L.append(df.round(fmt).to_markdown(index=index)); return df


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o): r = max(r, (len(p) - i) * p[j]); a[j] = min(1, r)
    return a


def ablation_pairs(name):
    p = R / f'r4_ablation_{name}_pred.csv'
    if not p.exists(): return None
    d = pd.read_csv(p); d['ape'] = 100 * (d.pred - d.soh_true).abs() / d.soh_true
    c = d.groupby(['variant', 'origin', 'seed', 'cell_id']).ape.mean().groupby(['variant', 'origin', 'cell_id']).mean(); rows = []
    for o in sorted(d.origin.unique()):
        x = c.xs(o, level='origin').unstack(0); fam = []
        for v in x.columns:
            if v == 'V1_main': continue
            dd = (x[v] - x['V1_main']).dropna(); fam.append(dict(dataset=name, origin=o, comparison=f'{v} - V1_main', n=len(dd), diff_pp=dd.mean(), p=stats.wilcoxon(dd).pvalue))
        fam = pd.DataFrame(fam); fam['p_holm'] = holm(fam.p); rows.append(fam)
    out = pd.concat(rows); out.to_csv(T / f'r4_ablation_{name}_pairs.csv', index=False); return out


def main():
    plan = json.loads((ROOT / 'PLAN_R4.json').read_text())
    L.append('# Round 4 report (Claude, cloud) — Luh-Blank, Sandia, Oxford, Kirkaldy\n')
    L.append('All choices were made on inner validation; every pre-declared arm is reported whatever its sign. Pre-declaration and amendments (with timestamps and the order in which results were seen) are in PLAN_R4.json; keys: ' + ', '.join(k for k in plan if k != 'trials') + '.\n')
    if 'R4A_compute_amendment' in plan: L.append('\n**Compute amendment** (' + plan['R4A_compute_amendment']['declared_utc'] + ' UTC): ' + plan['R4A_compute_amendment']['text'] + '\n')
    L.append('\n## 1. Luh-Blank: origin-specific re-search, refits and the hybrid (104 test cells)\n')
    tab('r4_summary.csv', 'Equal-cell capacity MAPE by method and origin (round-3 references + round-4)')
    tab('r4_pairs.csv', 'Paired tests vs ARIMA and vs round-3 best combination (Holm within family)'); tab('r4_hybrid_vs_tuned_learned_posthoc.csv', 'Post hoc: hybrid vs origin-specifically re-tuned learned models (Holm within these 8)')
    tab('r4_g6_arima_weight.csv', 'Validation-chosen ARIMA weight w*ARIMA+(1-w)*hybrid (G6)'); tab('r4_g6_weights.csv', 'G6 chosen weights')
    L.append('\n## 2. Why it works: ablations (R4F)\n')
    tab('r4_ablation_luh_summary.csv', 'Luh ablation summary', index=False); ablation_pairs('luh'); tab('r4_ablation_luh_pairs.csv', 'Luh ablations vs V1 (Holm within origin)')
    tab('r4_ablation_snl_summary.csv', 'Sandia ablation summary'); ablation_pairs('snl'); tab('r4_ablation_snl_pairs.csv', 'Sandia ablations vs V1')
    tab('r4_g3_noschedule_summary.csv', 'Hybrid without the test schedule (checkup index only, G3)'); tab('r4_g3_vs_main_pairs.csv', 'G3 minus main hybrid (paired, per origin)')
    L.append('\n## 3. Sandia (80 cells)\n'); tab('r4_hybrid_snl_pairs.csv', 'Sandia hybrid paired tests')
    L.append('\n## 4. New cell formats without target data\n')
    tab('r4_transfer_summary.csv', 'Kirkaldy zero-target transfer'); tab('r4_transfer_pairs.csv', 'Kirkaldy paired tests')
    tab('r4_transfer_ablation_summary.csv', 'Kirkaldy transfer by source set'); tab('r4_transfer_ablation_pairs.csv', 'Kirkaldy transfer source ablation')
    tab('r4_g1_oxford_summary.csv', 'Oxford frozen-design confirmation (no Oxford data used, G1)'); tab('r4_g1_oxford_by_horizon.csv', 'Oxford error by horizon bucket'); tab('r4_g1_oxford_pairs.csv', 'Oxford paired tests (n=8; minimum attainable p = 0.0078)')
    L.append('\n## 5. Pooling public datasets (R4C)\n'); tab('r4_pooled_summary.csv', 'Pooled vs source-only history-only models'); tab('r4_pooled_pairs.csv', 'Pooled vs source-only paired tests')
    L.append('\n## 6. Foundation-model baseline fine-tuned on sparse checkups\n'); tab('r4_chronos_ft_summary.csv', 'Fine-tuned Chronos-Bolt-small (decoder fine-tuned, encoder frozen; checkpoint by validation)'); tab('r4_chronos_ft_pairs.csv', 'Fine-tuned Chronos paired tests'); tab('r4_chronos_ft_snl_full_lr1e-5_selected.csv', 'G7 full fine-tune (lr 1e-5) selected checkpoints')
    L.append('\n## 7. Uncertainty and decisions\n')
    tab('r4_hybrid_conformal_luh.csv', 'Luh hybrid split-conformal (independent calibration cells)'); tab('r4_g5_mondrian_overall.csv', 'Luh group-conditional (regime) conformal, overall'); tab('r4_g5_mondrian_by_regime.csv', 'Luh group-conditional conformal by regime')
    tab('r4_hybrid_conformal_snl.csv', 'Sandia hybrid CV+-style intervals'); tab('r4_decisions_luh.csv', 'Luh decision metrics (common script)'); tab('r4_decisions_snl_hybrid.csv', 'Sandia hybrid decision metrics')
    L.append('\n## 8. Robustness\n'); tab('r4_g4_robust_summary.csv', 'Noise / dropped-checkup robustness at origin 20 (round-3 perturbations)'); tab('r4_g4_robust_pairs.csv', 'Hybrid minus ARIMA under perturbation'); tab('r4_g2_knee.csv', 'Knee vs non-knee cells')
    L.append('\n## Pending / not available\n' + ('\n'.join(f'- {p}' for p in pending) if pending else '- none'))
    (ROOT / 'REPORT_R4.md').write_text('\n'.join(L)); print('report written; pending:', pending)


if __name__ == '__main__':
    main()
