"""Create a fresh fitting directory, preserving delivered evidence and accepted records."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New directory that does not already exist')
    args = parser.parse_args()
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        raise FileExistsError('Use a new directory; existing outputs are never cleared: ' + str(target))
    if not target.parent.is_dir():
        raise FileNotFoundError('The output parent must already exist: ' + str(target.parent))
    target.mkdir()
    for name in ['audit', 'results', 'tables', 'logs', 'checkpoints', 'figures']:
        (target / name).mkdir()
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, target / p.name)
    for name in ['PLAN.json', 'requirements-recorded.txt', 'REPORT_TEMPLATE.md']:
        shutil.copy2(ROOT / name, target / name)
    for name in ['inputs', 'trainer']:
        shutil.copytree(ROOT / name, target / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.partial'))
    for p in (ROOT / 'audit').iterdir():
        keep = p.name.startswith(('prior_', 'raw_', 'USER_', 'legacy_')) or p.name in {
            'cell_partitions.csv', 'chronos_tiny_pinned_source.json'}
        if keep:
            if p.is_dir():
                shutil.copytree(p, target / 'audit' / p.name)
            else:
                shutil.copy2(p, target / 'audit' / p.name)
    # These are independent per-cell prefix baselines, not population-trained fits.
    # They are explicitly reused; all new tuned/secondary model jobs start fresh.
    for p in (ROOT / 'results').glob('reused_*.csv'):
        shutil.copy2(p, target / 'results' / p.name)
    for name in ['frozen_clean_secondary_summary.csv', 'raw_capacity_check.csv',
                 'legacy_Re0_actual_contributors.csv', 'legacy_Re_history_actual_contributors.csv',
                 'asynchronous_stream_checklist.csv']:
        shutil.copy2(ROOT / 'tables' / name, target / 'tables' / name)
    for p in (ROOT / 'tables').glob('earlier_*.csv'):
        shutil.copy2(p, target / 'tables' / p.name)
    if (ROOT / 'earlier_round').is_dir():
        shutil.copytree(ROOT / 'earlier_round', target / 'earlier_round')
    records = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in target.rglob('*') if p.is_file()}
    (target / 'audit/reproduction_setup.json').write_text(json.dumps({
        'source': str(ROOT), 'output': str(target), 'files': records,
        'policy': 'Fresh learned-model fitting. Independent per-cell baseline predictions and raw audit evidence reused explicitly. No accepted job markers or checkpoints copied.'}, indent=2), encoding='utf8')
    print('FRESH_REPRODUCTION_DIRECTORY', target)


if __name__ == '__main__':
    main()
