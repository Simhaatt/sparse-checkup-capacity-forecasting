#!/bin/bash
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/r4
while ! grep -q AB_DONE logs/ab.log; do sleep 60; done
python3 r4_pooled.py --workers 2 > logs/pooled.log 2>&1
echo POOLED_DONE >> logs/pooled.log
