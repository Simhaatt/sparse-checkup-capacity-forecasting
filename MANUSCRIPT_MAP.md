# Manuscript → code → files

`L1` means the item is regenerated and checked by `python reproduce/level1.py`. All tables below are in `results/tables/main_study/` unless another folder is given.

| Manuscript item | Table(s) | Produced by | L1 |
|---|---|---|---|
| Table 1, dataset cohorts | `data/processed/*`, `results/tables/sandia/origin*/partitions.csv` | preprocessing scripts; `r3.cohort` | – |
| Table 2, Fig. 3: primary accuracy, Holm families | `r4_summary.csv`, `r4_pairs.csv` | `r4_analyse.py` (original, run unchanged) | ✓ |
| Sect. 4.1 earlier benchmark | `results/tables/initial_benchmark/*`, `r4_summary.csv` | stage 1 | ✓ (values) |
| Sect. 4.4 re-tuned learners (post hoc) | `r4_hybrid_vs_tuned_learned_posthoc.csv` | `summarise_from_predictions.py` | ✓ |
| Sect. 4.4 fine-tuned Chronos | `r4_chronos_ft_summary.csv`, `r4_chronos_ft_pairs.csv` | `r4_chronos_ft.py`, `r4_post.py chronos` | deposited |
| Fig. 4, Sect. 4.3 horizon and regime | `r4_hybrid_error_by_horizon.csv`, `r4_by_age_type.csv` | `summarise_from_predictions.py` | ✓ |
| Sect. 4.5, Table C.3 ablations | `r4_ablation_luh_summary.csv`, `r4_ablation_luh_pairs.csv` (Sandia: `r4_ablation_snl_*`) | `r4_ablate.py` | ✓ (Luh–Blank) |
| Sect. 4.5 no-schedule sensitivity | `r4_g3_noschedule_summary.csv`, `r4_g3_vs_main_pairs.csv` | `r4_g.py g3` | ✓ (Luh–Blank) |
| Table 3, Sect. 4.6: Sandia | `results/tables/sandia/*`, `r4_hybrid_snl_*`, `r4_chronos_ft_summary.csv` | stage 2, `r4_hybrid_snl.py` | aggregate only |
| Fig. 5, Table C.4: 21700 transfer | `r4_transfer_summary.csv`, `r4_transfer_pairs.csv`, `r4_transfer_ablation_*` | `r4_transfer.py`, `r4_transfer_ablate.py` | ✓ |
| Fig. 5: pouch transfer | `r4_g1_oxford_summary.csv`, `r4_g1_oxford_pairs.csv` | `r4_g.py g1` | ✓ |
| Fig. 6a, Table D.1: split-conformal intervals | `r4_hybrid_conformal_luh.csv` | `r4_uncert.py` | ✓ |
| Fig. 6b: regime-stratified intervals | `r4_g5_mondrian_by_regime.csv`, `r4_g5_mondrian_overall.csv` | `r4_g.py g5` (original, run unchanged) | ✓ |
| Fig. 6c: Sandia intervals | `r4_hybrid_conformal_snl.csv` | `r4_uncert.py` | aggregate only |
| Fig. 6d, Table D.2: crossing decisions | `r4_decisions_luh_r3rule.csv` (Sandia: `r4_decisions_snl_hybrid.csv`) | `r4_decisions_v2.py` (original, run unchanged) | ✓ |
| Sect. 4.9, Table D.3: robustness and knees | `r4_g4_robust_summary.csv`, `r4_g4_robust_pairs.csv`, `r4_g2_knee.csv` | `r4_g.py g4`, `r4_g.py g2` | ✓ (g4) |
| Mixing-weight sensitivity | `r4_g6_arima_weight.csv`, `r4_g6_weights.csv` | `r4_g.py g6` | deposited |
| Pooling experiment | `r4_pooled_summary.csv`, `r4_pooled_pairs.csv` | `r4_pooled.py`, `r4_post.py pooled` | deposited |
| Table B.2: fold selections | `r4_hybrid_selected.csv` | `r4_hybrid.py` | deposited |
| Appendix A, Fig. A.1: leakage audit | `leakage_audit/tables/*` | `leakage_audit/*.py` | – |
| Appendix E: plan chronology | `experiments/stage3_main_study/PLAN_R4.json`, `logs/` | `r4_plan.py` | – |
| Figs. 1–6, A.1 | `paper_figures/data_used/*` | `paper_figures/scripts/*` | ✓ (plotted values) |
