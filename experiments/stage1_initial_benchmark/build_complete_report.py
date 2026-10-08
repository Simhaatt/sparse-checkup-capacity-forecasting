"""Final MD only after full numerical verification and actual visual inspection."""
from r3 import *
from write_report import table

def main():
    verification=json.loads((ROOT/'audit/completion_verification.json').read_text());visual=json.loads((ROOT/'audit/visual_qa.json').read_text())
    assert verification['passed'] and visual['passed'],'Do not publish completed status from partial evidence.'
    manifest=pd.read_csv(ROOT/'tables/figure_manifest.csv')
    assert set(visual['inspected_pngs'])==set(manifest.png)
    plan=prepare();text=(ROOT/'REPORT_TEMPLATE.md').read_text(encoding='utf8')+'\n'
    lines=[]
    for line in text.splitlines():
        if line=='| Requested item | Evidence in this report |':line='| Requested item | Status | Evidence in this report |'
        elif line=='| --- | --- |':line='| --- | --- | --- |'
        elif line.startswith('| ') and ('Earlier request:' in line or line[2:3].isdigit()):
            fields=line.split('|');label=fields[1].strip()
            status='done; aligned diagnostic unavailable by user decision' if label.startswith('6.') else 'done'
            line='| '+label+' | '+status+' | '+fields[2].strip()+' |'
        lines.append(line)
    text='\n'.join(lines)+'\n'
    names=['earlier_transfer_capacity_summary.csv','earlier_transfer_capacity_pairs.csv','earlier_quiet_capacity_resources.csv','capacity_summary.csv','frozen_clean_secondary_summary.csv','capacity_by_type.csv','paired_tests.csv','HOLM_FAMILIES.csv','best_neural_by_fold_origin.csv','selector_rules.csv','resistance_arm_ranks.csv','conformal_overall.csv','conformal_by_type.csv','native_chronos_coverage.csv','decision_summary.csv','learning_summary.csv','learning_pairs.csv','learning_effective_data.csv','oxford_summary.csv','oxford_pairs.csv','robustness_summary.csv','robustness_pairs.csv','robustness_arima_statuses.csv','historical_macro_summary.csv','aligned_macro_unavailable.csv','raw_capacity_check.csv','legacy_Re0_actual_contributors.csv','legacy_Re_history_actual_contributors.csv','asynchronous_stream_checklist.csv']
    for name in names:
        p=ROOT/'tables'/name;assert p.exists(),name;text+=f'### {name}\n\n[Full-precision CSV](tables/{name})\n\n'+table(pd.read_csv(p))+'\n\n'
    for p in [ROOT/'tables/fold_selected_settings.csv',ROOT/'results/semiempirical_runs_v3.csv',ROOT/'RESULTS_SUMMARY.csv']:
        text+=f'### {p.name}\n\n[CSV]({p.relative_to(ROOT).as_posix()})\n\n'+table(pd.read_csv(p))+'\n\n'
    for name in ['quiet_primary_resources.csv','search_training_costs.csv','validation_mean_frontier.csv']:
        p=ROOT/'tables'/name;assert p.exists(),name
        text+=f'### {name}\n\n[Full-precision CSV](tables/{name})\n\n'+table(pd.read_csv(p))+'\n\n'
    text+='## Research figures\n\n'
    for row in manifest.itertuples():text+=f'### {row.figure}\n\n![{row.figure}]({row.png})\n\n[PDF]({row.pdf}); sources: `{row.sources}`.\n\n'
    text+='## Full table catalogue\n\n'
    catalogue=[]
    for p in sorted((ROOT/'tables').glob('*.csv')):
        frame=pd.read_csv(p,nrows=0);catalogue.append(dict(table=p.name,columns=', '.join(frame.columns),source=f'tables/{p.name}'))
        text+=f'- [{p.name}](tables/{p.name}).\n'
    pd.DataFrame(catalogue).to_csv(ROOT/'audit/table_catalogue.csv',index=False)
    text+='''
## Reproduce in a separate extracted directory

```powershell
# From the delivered directory, create a fresh directory first:
python -X utf8 prepare_reproduction.py --output 'C:/Users/hp/Desktop/JES_round3_rerun'
Set-Location 'C:/Users/hp/Desktop/JES_round3_rerun'
python -m pip install -r requirements-recorded.txt
python -X utf8 aligned_resistance_v2.py
python -X utf8 operating_metadata.py
python -X utf8 run_models.py --smoke
python -X utf8 run_models.py
python -X utf8 semiempirical_v3.py
python -X utf8 summary_current.py
python -X utf8 chronos_tiny.py
# Download pinned small weights for Oxford; also reproduce the source baseline:
python -X utf8 rebuild_chronos_small.py
python -X utf8 native_chronos_coverage.py
python -X utf8 leaky_macro_diagnostic.py
python -X utf8 analyse_round3.py
python -X utf8 learning_curve.py
python -X utf8 oxford_models.py
python -X utf8 robustness.py prepare
python -X utf8 robustness.py arima
python -X utf8 robustness.py models
python -X utf8 analyse_extras.py
# After own fitting jobs have exited:
python -X utf8 quiet_resources.py
python -X utf8 make_plots.py
python -X utf8 verify_round3.py
# Actual image inspection and visual_qa.json are required before final report:
python -X utf8 build_complete_report.py
```

Completed model drivers resume by jobID and checkpoint hash; use the fresh directory prepared above for fitting reruns. It deliberately reuses the independently fitted per-cell baseline predictions and raw provenance receipts, but copies no learned-model checkpoints or accepted fitting markers. Preserve earlier evidence. Raw files/external weights use original source manifests and pinned revisions; the exact environment is recorded under `audit/environment_*.json`. `PLAN.json` contains the declared settings, partition, scaling, information and aggregation rules, and input source hashes.

## Verification and sources

'''
    text+=f"Full numerical verification: **{len(verification['checks'])} checks passed**. [Checks CSV](audit/completion_checks.csv), [JSON](audit/completion_verification.json), [visual inspection](audit/visual_qa.json). These do not turn retrospective conditional experiments into field guarantees.\n\n"
    text+='- [Luh–Blank data descriptor](https://www.nature.com/articles/s41597-024-03831-x).\n- [Actual result-v2 dataset](https://www.radar-service.eu/radar/en/dataset/CoNQplSNoVeXExyV).\n- [Oxford dataset/licence record](https://ora.ox.ac.uk/objects/uuid%3A03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac), with the Birkl 2017 thesis citation required by README.\n- [Chronos code](https://github.com/amazon-science/chronos-forecasting) and [small official model](https://huggingface.co/amazon/chronos-bolt-small).\n- [Schmalstieg institutional record](https://publications.rwth-aachen.de/record/235563/), inspiration rather than coefficient reproduction.\n'
    out=ROOT/'REPORT_COMPLETE.md';out.write_text(text,encoding='utf8');(ROOT/'REPORT.md').write_text(text,encoding='utf8');print('COMPLETED_REPORT_WRITTEN',out,flush=True)
if __name__=='__main__':main()
