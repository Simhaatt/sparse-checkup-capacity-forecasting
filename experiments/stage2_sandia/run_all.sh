#!/bin/bash
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/sandia/bench
for o in 5 8; do
  export SNL_ORIGIN=$o
  python3 snl_bench.py prepare && python3 snl_bench.py controls && python3 snl_bench.py search && python3 snl_bench.py final && python3 snl_bench.py analyse
done
echo ALLDONE
