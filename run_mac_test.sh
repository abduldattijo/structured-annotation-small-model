#!/bin/bash
# Tries batch sizes 4, 2, 1 and stops at the first one that finishes.
cd "$(dirname "$0")"
for b in 4 2 1; do
  PYTHONUNBUFFERED=1 .venv/bin/python mac_test.py $b > results/mac_test_batch$b.log 2>&1
  if [ -f results/mac_test_batch$b.json ]; then echo "done batch $b" > results/mac_test_status.txt; exit 0; fi
  echo "batch $b failed" >> results/mac_test_status.txt
done
echo "all failed" >> results/mac_test_status.txt
