#!/bin/bash
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/r4
set -e
for o in 10 20 15; do
  python3 r4_models.py search --origin $o --workers 2
  python3 r4_models.py final --origin $o --workers 2
  echo "ORIGIN_DONE $o"
done
python3 r4_pooled.py --workers 2
echo R4_ALL_DONE
