"""Current round3 status, with explicit incomplete gates and CSV-backed tables."""
from r3 import *

def table(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'unavailable' if pd.isna(v) else f'{v:.6g}'
        return str(v).replace('|',r'\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(fmt,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
def count(name):
    p=ROOT/'results'/name;return len(pd.read_csv(p)) if p.exists() else 0
def passed(name):
    p=ROOT/'audit'/name;return p.exists() and json.loads(p.read_text()).get('passed',False)
def main():
    plan=prepare();metadata('report')
    text='''# JES evidence round3: capacity-only experiment report

Client date:7 October2026, Asia/Calcutta. **This is a progress report. The full round is not complete.** Existing outputs and original manuscripts are preserved. No manuscript drafting/editing, GitHub push, DOI creation or journal contact.

The current human instructions are preserved in [USER_ROUND3_REQUEST.txt](audit/USER_ROUND3_REQUEST.txt). Their protocol supersedes earlier exploratory implementation choices. The previous100-fit, one-fold search used a different validation split and a larger compact family; it was stopped and preserved in `C:/Users/hp/Desktop/pinn/JES_capacity_retuning`. It is not the primary round3search.

## Status of every requested item

| Item | Status | Evidence/reason |
|---|---|---|
'''
    rows=[
        [1,'partial',f"{count('search_runs.csv')}/500search trials;{count('final_runs.csv')}/375final/replay fits; quiet all-trial/winner costs and frontier pending."],
        [2,'partial','Four prespecified variants. Full/no-k/history-only fitting follows tuning. All six descriptors are constant within every source cell, so static/no-k inputs are exactly identical; static results reuse the identical fitted predictor with explicit aliases, not a significance/equivalence claim.'],
        [3,'done','Only the prespecified variants; prior global descriptor ranking is retired. All new model fitting/selection is confined to training/inner-validation roles.'],
        [4,'done','Three raw EOC reconstructions match delivered Q exactly; Q retained by the EIS join; cumulative minimum is past-only. See raw_capacity_check.csv and prior_raw_capacity_receipt.json.'],
        [5,'done','All228source EIS files obtained and CRC verified; completed-event timestamp groups; backward14-day join. Exactly228matchedcapacity rows, one per cell. No filling of missing labels, cummax or whole-cell clipping.'],
        [6,'partial',f"{count('resistance_runs.csv')}/300B/Cfits; A reuses clean full fits at20/10. Aligned post-origin macro-MAPE is unavailable and explicitly waived by the user; no future aligned Re labels exist under the fixed rule."],
        [7,'done','Twelve CSV-backed practitioner checks, including sweep IDs, temporal availability, missingness, contributor traces, separate endpoints and physical usage-counter definitions.'],
        [8,'partial','Requires new fold-tuned predictions: best neural chosen on inner validation, equal triple and pairwise means, regime/origin selector. Saved frozen-data averages remain secondary.'],
        [9,'partial','Independent inner calibration cells reserved;80/90horizon intervals and ageing-type coverage/width pending final predictions.'],
        [10,'partial','Frozen capacity-only metrics already exist; new tuned last-step/bias/crossing/horizon tables pending final predictions.'],
        [11,'partial','Official pinned Chronos-Bolt small andtiny forecasts on104cells at20/10/15 are complete. Fold-role/native-quantile coverage tables pending. Nativep10/p90central80%, notnative90%; index-time/irregular-spacing caveat applies.'],
        [12,'done','Fifteen training-only population fits, fivefolds×seeds42/52/62; fitted calendar/cycle exponents and Arrhenius coefficients; exactly one measured-prefix cell scale. Corrected nominalFEC uses completed operating recharge counts times protocolDoD. v1/v2retained as exploratory; v3is current.'],
        [13,'partial','Retained as requested: tunedXGB/bestneural25/50/100training fractions with3random subset repeats, vsARIMA. Waits for fold winners.'],
        ['Earlier extra:Oxford','partial','Official8cell checkup data downloaded/audited; independent six-fit/one-validation/one-test cell evaluation pending.'],
        ['Earlier extra:robustness','partial','Prefix-noise/dropout scenarios retained; model refitting/inference tests pending.']]
    for item,status,reason in rows:text+=f'| {item} | {status} | {reason} |\n'
    text+='''
## Protocol and statistical units

- Same104scored source cells and five supplied grouped cell folds. Model-selection validation contains round(15% of all outer-training cells), sampled among scoreable cells with seed1000+fold. It is27cells per fold.
- Twenty **additional**, disjoint inner calibration cells are withheld for honest split-conformal calibration. They are not early-stopping/configuration-selection cells. This reduces fitting to135/136cells per fold and must be distinguished from the older frozen benchmark's larger fitting pool.
- Twenty random configurations per family/fold; compact LSTM hidden16–64, one layer; large LSTM hidden128–256, one/two layers; declared GRU/tree spaces in PLAN.json. The same draws are reused across folds. Matching trial count does not mean matching CPU hours.
- Teacher-forced pretraining followed by corrected differentiable recursive fine-tuning. Both validation checkpoint stages use equal-cell capacity errors. Main configuration selection is recursive validation Q-MAPE, never outer-test error.
- Final seeds42/52/62. Primary aggregation is mean stepAPE within seed/cell, mean across seeds within each cell, then equal cells. Bootstrap10,000draws, seed20261005. Paired Wilcoxon/explicit Holm families must be reported with the finished tables. Non-significance is not equivalence.
- Current descriptor rowi is excluded. After the origin, Q is self-fed. Full F7 is conditional on the benchmark operating schedule; no-k/static/history-only omit the future operation counter.
- Origins10/15 are separately labelled replays of the origin20-selected structures with L clipped to the prefix; they are not additional origin-specific searches.

## Raw resistance availability and corrections to earlier rounds

`seq_nr` is a frequency-point index reused at later checkups, not a sweep identifier. The original preprocessing could group the same frequency across dates and then assign artificial ordinals to these groups. Correct raw spectra group the29frequency points sharing a completed-event timestamp.

The first rebuilding attempt also used seq_nr and produced no valid frequency-diverse groups. It is rejected in audit/aligned_v1_rejected.json; its source and reconstructed rejected outputs are preserved. Only **aligned_re_v2.csv** and **raw_eis_sweeps_v2.csv** are scientific alignment inputs.

The prescribed backward14-day rule yields exactly one matching checkup per source cell, before origin10. Later checkups do not have a recent earlier EIS sweep. All228rawfiles were obtained, so this is genuine sparse temporal availability rather than failed acquisition. No cell has full aligned coverage of its capacity trajectory.

At origins10/20, C therefore contains initial resistance information and missingness. It cannot support resistance forecasting. Missing raw labels remain NaN; numerical zero is used only as an input sentinel with mask0. Re0 becomes available only after the first causal aligned observation. After each forecast origin, Re/Re0 are held from pre-origin information and mask0, including simulated training rollouts. B/C have identical input dimensions: B receives a prefix availability mask as well, so the network itself does not change between B/C.

The user explicitly chose **keep the fixed14-day rule and report the aligned macro-MAPE diagnostic as unavailable**. See audit/USER_ALIGNMENT_DECISION.json. Do not manufacture aligned resistance targets, widen the tolerance or join forward.

Earlier maximal-group-timestamp lookahead diagnostics need qualification: a frequency group spanning later dates does not by itself prove which timestamps contributed to a three-point mean. Whole-cell clipping, incorrect grouping and Rct-like full-group operations are separate problems. The new report must not repeat unverified numeric lookahead claims. Capacity-only conclusions are independent of those excluded Re-derived inputs.

The explicit three-cell trace found no post-origin10/20 contribution to the raw Re0 means. That corrects the earlier blanket inference about Re0. The actual Re-history contributor table separately tests the later values selected by the ordinal join. This does not validate the older preprocessing: invalid frequency grouping, whole-cell transformations and full-group Rct-like computations remain distinct defects.

The source profile num_cycles_op counter increments on partial discharge events. It is not a true full-cycle/FEC count. The k_exp identifier is retained unchanged for the requested F7benchmark, but should be described as an operation counter. The semi-empirical v2 exposure counts complete operating recharges, multiplied by the fixed nominal protocolDoD. It excludes diagnostic cycles and is a nominal controlled-cycle approximation, not realizedAhthroughput or a mechanistic identification.

## Current receipts and full-precision tables

All printed numbers use six significant figures; CSVs retain their original precision.

'''
    for file,folder in [('raw_capacity_check.csv','tables'),('eis_group_identifier_check.csv','audit'),('aligned_re_coverage_v2.csv','tables'),('legacy_Re0_actual_contributors.csv','tables'),('legacy_Re_history_actual_contributors.csv','tables'),('operating_counter_vs_fec.csv','tables'),('semiempirical_runs_v3.csv','results'),('completed_baseline_pairs.csv','tables'),('asynchronous_stream_checklist.csv','tables')]:
        p=ROOT/folder/file
        if not p.exists():continue
        text+=f'### {file}\n\n[Full-precision CSV]({folder}/{file})\n\n'+table(pd.read_csv(p))+'\n\n'
    if (ROOT/'RESULTS_SUMMARY.csv').exists():text+='### Current results summary\n\n[Full-precision RESULTS_SUMMARY.csv](RESULTS_SUMMARY.csv)\n\n'+table(pd.read_csv(ROOT/'RESULTS_SUMMARY.csv'))+'\n\n'
    text+='''## Reproduction commands and completion gates

```powershell
Set-Location 'C:\\Users\\hp\\Desktop\\pinn\\JES_review_extensions\\round3'
python -X utf8 aligned_resistance_v2.py
python -X utf8 operating_metadata.py
python -X utf8 run_models.py --smoke
python -X utf8 run_models.py
python -X utf8 semiempirical_v3.py
python -X utf8 summary_current.py
python -X utf8 write_report.py
```

Model drivers resume accepted jobIDs and validate their checkpoint hashes. Raw acquisition resumes cached CRC-verified cell files. Do not overwrite/delete prior completed studies. Quiet resource timing must be performed after the fitting processes exit; loaded trial times are exploratory.

The remaining analysis/learning/Oxford/robustness drivers will be listed when implemented and verified. This report does not claim that pending scripts or green smoke tests prove completion of the full scientific objective. Final REPORT.md/RESULTS_SUMMARY.csv require completed experiment grids, fold/cell/seed/row integrity, proper statistical families, figure inspection, and an item-by-item completion audit.
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf8');print('ROUND3_PROGRESS_REPORT',ROOT/'REPORT.md',flush=True)
if __name__=='__main__':main()
