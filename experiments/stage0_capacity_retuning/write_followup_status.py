"""Evidence-backed progress MD; never marks the expanded goal complete from a running fit."""
from study import *
import subprocess
OUT=ROOT.parent/'JES_CAPACITY_FOLLOWUP_STATUS.md'

def receipt(name):
    p=ROOT/'audit'/name
    return p.exists() and json.loads(p.read_text()).get('passed',False)
def count(name):
    p=ROOT/'results'/name
    return len(pd.read_csv(p)) if p.exists() else 0
def table(d):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'not available' if pd.isna(v) else f'{v:.5g}'
        return str(v).replace('|',r'\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(fmt,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
def main():
    lines=['# JES expanded capacity follow-up: current verified status','',
        'Client date:6 October2026 (Asia/Calcutta). This is a progress MD. The expanded goal is active; the retuning and later learning/robustness/Oxford evaluations are not yet complete. Earlier final MD remains the completed previous study, not proof that the new work has finished.','',
        'All four earlier requests plus all ten new requests are retained, including the second dataset and robustness items originally labelled optional. No manuscript, repository push, submission or DOI release.','',
        '## Requirement-by-requirement status','',
        '| Requirement | Current evidence/status |','|---|---|',
        f'| Clean retuning | {count("search_runs.csv")}/500 search fits; {count("final_runs.csv")}/375 final/replay/ablation fits. Twenty fresh settings per five families per five outer folds. |',
        f'| No-future-count ablation | Smoke invariant passed:{receipt("smoke_receipt.json")}; full prefix-fixed/history-only fits queued within the new fold-specific protocol. |',
        '| Feature ablation | Old global feature ranking retired; only prespecified reduced-information fits inside the new training partitions are active. Full fits pending. |',
        f'| Raw capacity | Three-cell direct reconstruction and future-tamper checks passed:{receipt("raw_capacity_receipt.json")}. |',
        f'| Average/selector | Prespecified saved-prediction average completed:{receipt("saved_analysis_receipt.json")}. Inner-only selector waits for new fitted inner predictions. |',
        '|80%/90% intervals | Independent calibration cells reserved; exact horizon-specific split-conformal analysis implemented. Full calibration/test predictions pending. |',
        f'| Decision metrics | Clean saved-predictor crossing/last-step/bias tables completed:{receipt("saved_analysis_receipt.json")}; retuned version pending. |',
        f'| Horizon curves | Saved clean-data PNG/PDF/CSV completed:{receipt("saved_analysis_receipt.json")}; retuned version pending. |',
        f'| Pretrained baseline | Pinned Chronos-Bolt small CPU inference completed:{receipt("chronos_inference_receipt.json")}, all104 cells at10/15/20. |',
        f'| Population ageing | Five fit-only population models and prefix-only cell calibrations completed:{receipt("semiempirical_receipt.json")}; future-Q invariance passed:{receipt("semiempirical_tamper.json")}. |',
        '| Learning curves |25%/50% subset-specific twenty-setting searches implemented;100% comes from main retuning. Additional1,000 searches and150 final fits pending. |',
        f'| Second source dataset | Official Oxford download/input audit completed:{receipt("oxford_input_receipt.json")}; eight-cell independent model evaluation pending. |',
        '| Robustness | Prefix measurement-noise and dropped-checkup experiments pending; original future truth must remain unchanged. |',
        '| Training cost | New fits record wall/CPU time, checkpoint size and inference time. Full cost/frontier tables and an idle-suite timing pass pending. |','',
        '## Frozen protocol and claims boundaries','',
        'Five fixed outer cell folds. Inside each, fit, hyperparameter-selection, interval-calibration and test cells are mutually disjoint. The calibration cells are never early-stopping/selection cells. Every family receives20 outcome-independent random settings; matching trial count does not imply matching CPU hours. Main tuning is at20 measured checkups. Origins10/15 replay these newly selected settings and are labelled accordingly.','',
        'Final fitting uses seeds42/52/62. At origin20, each selected configuration is retrained with scheduled F7+Q history, prefix-frozen six descriptors+Q history and Q history alone. Fixed descriptors are frozen at the first measured checkup; no cycle/time descriptor enters the reduced-information model. Chronological ordering uses timestamps, not future cycle values. Input/output scalers fit training cells only.','',
        'The learning-curve design tunes independently using only the25% or50% fitting subset; it does not reuse full-data hyperparameter outcomes to claim small-campaign data efficiency. Selection/calibration sets remain fixed and their size must be disclosed alongside the fitting-cell fraction.','',
        'Split-conformal radii are calibrated per horizon using absolute-SOH residuals and rank ceil((n+1)*nominal). Insufficient horizon samples require an infinite bound; finite interval coverage/width and finite fraction must be reported. No simultaneous trajectory coverage or unconditional field guarantee is claimed.','',
        'The saved average uses equal ARIMA/XGBoost/large-LSTM weights chosen before analysis. Three trained-seed predictions are averaged before scoring; this differs from the prior MD\'s seed-wise error average. Lower point error is not automatic statistical superiority.','',
        '## Completed numerical additions','']
    for name in ['raw_capacity_check.csv','saved_clean_combination_summary.csv','saved_clean_combination_pairs.csv','saved_clean_decision_summary.csv','chronos_cost.csv','oxford_input_audit.csv']:
        p=ROOT/'tables'/name
        if p.exists():
            frame=pd.read_csv(p)
            if name=='saved_clean_combination_summary.csv':frame=frame[frame.label.isin(['ARIMA_AICc','xgb_tuned_Q_only','Manual_Q_only','equal_ARIMA_XGB_largeLSTM','ChronosBolt_small_zero'])]
            lines.extend([f'### {name}','',table(frame),''])
    lines+=['## Important provenance and physical-model qualifications','',
        'Raw Q on the three reconstructed cells is cap_aged_est_Ah retained from the EOC left table. The EIS nearest join does not substitute Q labels. The filtered Q cumulative minimum exactly reproduces the delivered Q; no whole-cell Q clipping is applied. This audit does not independently reconstruct the upstream electrochemical capacity estimator or establish every raw cell\'s provenance.','',
        'The population model is an empirical additive calendar/cycle law with square-root elapsed-day/cycle terms and exponential temperature/SOC/C-rate modifiers. It is Schmalstieg-inspired, not a reproduction of published coefficients. Cyclic/profile SOC bounds are unavailable in the supplied schema; the nominal0.5 value is disclosed. Predictions condition on lagged benchmark exposure coordinates.','',
        'Oxford capacities are raw diagnostic1C discharge endpoint charge differences, converted frommAh toAh. No cumulative minimum is applied. One exactly duplicated full discharge record inCell1 was removed. Original timestamps are MATLAB serial days despite the README\'s generic seconds description; relative time is derived from the observed units. Exact chemistry is not inferred solely from this README.','',
        'Chronos-Bolt small uses the locally pinned official model revision and native0.5-quantile forecasts. No fine-tuning; positive clipping only. Pretraining dataset overlap was not independently audited.','',
        '## Files and primary sources','',
        f'- Workspace: `{ROOT}`.',
        f'- Frozen plan: [plan.json]({ROOT.as_posix()}/plan.json).',
        f'- Horizon plot: [PNG]({ROOT.as_posix()}/tables/saved_clean_error_by_horizon.png), [PDF]({ROOT.as_posix()}/tables/saved_clean_error_by_horizon.pdf).',
        f'- Exact active scope: [EXPANDED_GOAL.md]({ROOT.as_posix()}/EXPANDED_GOAL.md).',
        '- [Amazon Chronos code](https://github.com/amazon-science/chronos-forecasting) and [official small model](https://huggingface.co/amazon/chronos-bolt-small).',
        '- [Oxford dataset and licence record](https://ora.ox.ac.uk/objects/uuid%3A03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac); cite the dataset and Birkl2017 thesis as requested by its README.',
        '- [Schmalstieg etal. institutional record](https://publications.rwth-aachen.de/record/235563/), DOI10.1016/j.jpowsour.2014.02.012.','']
    OUT.write_text('\n'.join(lines),encoding='utf8');print('FOLLOWUP_STATUS_MD',OUT,flush=True)
if __name__=='__main__':main()
