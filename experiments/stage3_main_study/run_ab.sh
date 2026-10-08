#!/bin/bash
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/r4
python3 r4_models.py search --workers 2 && python3 r4_models.py final --workers 2
echo AB_DONE
