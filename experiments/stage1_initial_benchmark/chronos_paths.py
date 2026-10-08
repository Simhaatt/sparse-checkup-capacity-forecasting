"""Resolve verified pinned small-model weights without requiring the old workspace."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def small_weights():
    record = json.loads((ROOT / 'audit/prior_chronos_download_receipt.json').read_text())
    candidates = [ROOT / 'inputs/chronos-bolt-small',
                  ROOT.parent.parent / 'JES_capacity_retuning/external/chronos-bolt-small']
    for folder in candidates:
        if all((folder / Path(item['file'].replace('\\', '/')).name).is_file()
               for item in record['files']):
            for item in record['files']:
                p = folder / Path(item['file'].replace('\\', '/')).name
                assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
            return folder
    raise FileNotFoundError('Run python -X utf8 rebuild_chronos_small.py to download the pinned small model.')
