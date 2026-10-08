"""Readable, timestamped partial handoff; never labels pending experiments complete."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent


def rows(relative):
    p = ROOT / relative
    if not p.exists():
        return []
    with p.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def count(name):
    return len(rows('results/' + name + '_runs.csv'))


def passed(name):
    p = ROOT / 'audit' / name
    return p.exists() and json.loads(p.read_text(encoding='utf8')).get('passed', False)


def table(relative):
    records = rows(relative)
    if not records:
        return ''
    def fmt(value):
        try:
            number = float(value)
            return 'unavailable' if math.isnan(number) else f'{number:.6g}'
        except (TypeError, ValueError):
            return str(value).replace('|', r'\|').replace('\n', '<br>')
    columns = list(records[0])
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for row in records]
    return '\n'.join(lines)


def main():
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    status = [
        ('1. Retuning', 'partial', f'{count("search")}/500 search and {count("final")}/375 core fits; quiet costs/frontiers and full verification pending.'),
        ('2. Information variants', 'partial', f'Core grid above; {count("static_alias")}/75 explicit static aliases. Paired analysis and final verification pending.'),
        ('3. Descriptor selection', 'done', 'Only predeclared variants; global feature ranking retired.'),
        ('4. Raw capacity', 'done', 'Three-cell exact reconstruction and past-only cumulative-minimum receipt passed.'),
        ('5. Raw EIS', 'done', 'All 228 files obtained; fixed backward 14-day alignment reconstructed and audited.'),
        ('6. A/B/C', 'partial', f'{count("resistance")}/300 B/C fits; A reuses core full models. Aligned post-origin joint-macro diagnostic unavailable under the user-approved fixed rule.'),
        ('7. Checklist', 'done', 'CSV practitioner checklist retained.'),
        ('8. Combinations/selector', 'partial', 'Implementation and selection-isolation checks passed; full real-data analysis queued.'),
        ('9. Conformal intervals', 'partial', 'Disjoint calibration cells reserved; math check passed; full calibration/coverage queued.'),
        ('10. Decisions/horizon', 'partial', 'New tuned-model tables and figures queued; older clean results preserved.'),
        ('11. Chronos small/tiny', 'done', 'Source median predictions and native 80% coverage completed; conformal coverage belongs to item 9.'),
        ('12. Population law', 'done', '15 training-only version-3 fits and causal prefix calibration completed.'),
        ('13. Learning curves', 'partial', f'{count("learning")}/180 subset fits; full-pool reference uses main models.'),
        ('Earlier second-dataset request', 'partial', f'{count("oxford")}/120 Oxford learned fits; raw input audited and evaluation queued.'),
        ('Earlier robustness request', 'partial', f'{count("robustness_arima")}/2600 ARIMA fits and {count("robustness_model")}/75 learned inference jobs; full comparison queued.'),
        ('Final verification/delivery', 'partial', 'Quiet measurements, numerical verification, six figure inspections, completed Markdown and verified ZIP remain required.'),
    ]
    text = '# JES evidence round 3: updated progress handoff\n\n'
    text += f'Updated {stamp}. **The full round is still running. This document does not claim final verification or completion.**\n\n'
    if passed('core_verification.json'):
        core=json.loads((ROOT/'audit/core_verification.json').read_text(encoding='utf8'))
        text += f"Completed core audit: **{len(core['checks']):,} checks passed**, covering all completed search/core fits and static aliases. [Core verification](audit/core_verification.json); [CSV score round-trip and unchanged winners](audit/validation_score_roundtrip_receipt.json). This is separate from the pending full-round completion audit.\n\n"
    text += '| Item | Status | Actual evidence / remaining work |\n| --- | --- | --- |\n'
    text += '\n'.join('| ' + ' | '.join(row) + ' |' for row in status) + '\n\n'
    text += ('The active sequence is: finish resistance fits; analyze the complete main grid; run learning curves, Oxford and learned-model robustness; analyze the extra studies; measure quiet resources; export and verify figures; perform actual visual inspection; generate the completed report and evidence archive. Accepted jobs resume by ID. Earlier outputs remain preserved.\n\n'
             'The following protocol text describes the intended full evidence handoff. The status table above governs what is actually complete at this timestamp.\n\n')
    template = (ROOT / 'REPORT_TEMPLATE.md').read_text(encoding='utf8')
    # Avoid a second document title and replace completion wording in this partial view.
    template = template[template.index('\n') + 1:]
    template = template.replace('## Completion and requested scope', '## Full requested scope')
    text += template + '\n'
    text += '## Verified results already available\n\n'
    for relative in ['RESULTS_SUMMARY.csv', 'tables/native_chronos_coverage.csv',
                     'tables/earlier_transfer_capacity_summary.csv', 'tables/earlier_transfer_capacity_pairs.csv',
                     'tables/historical_macro_summary.csv', 'tables/aligned_macro_unavailable.csv',
                     'tables/raw_capacity_check.csv', 'tables/asynchronous_stream_checklist.csv']:
        rendered = table(relative)
        if rendered:
            text += f'### {Path(relative).name}\n\n[Full-precision CSV]({relative})\n\n{rendered}\n\n'
    text += ('## Completion receipts and reproduction\n\n'
             'Individual completed-study receipts are in `audit/`. The final `completion_verification.json`, actual `visual_qa.json` and `package_receipt.json` are the completion gates. Their absence means this progress report is not a completed handoff.\n\n'
             'The full command sequence is retained in `build_complete_report.py` and will be printed in the final report. `prepare_reproduction.py --output NEW_DIRECTORY` creates a separate fresh fitting workspace without accepted model-job markers or checkpoints; it explicitly reuses independent per-cell baselines and raw provenance evidence. Do not rerun fitting inside the preserved delivered results.\n')
    (ROOT / 'REPORT.md').write_text(text, encoding='utf8')
    (ROOT / 'audit/progress_report_receipt.json').write_text(json.dumps({
        'timestamp': stamp, 'status': 'partial', 'report_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'counts': {k: count(k) for k in ['search', 'final', 'static_alias', 'resistance', 'learning', 'oxford', 'robustness_arima', 'robustness_model']},
        'passed_completed_ARIMA': passed('robustness_arima_receipt.json'),
        'completed_report_generated': False}, indent=2), encoding='utf8')
    print('UPDATED_PARTIAL_REPORT', ROOT / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
