#!/bin/bash
# Run the fast pipeline on the evaluation subset at several settings.
# Usage: bash eval/sweep.sh   (from the repo root)
PY=~/miniconda3/envs/finevideo/bin/python
IN=/home/beast/Data/Arslan/finevideo_eval/subset
for rate in 1 2 3 5 10; do
  $PY run.py --input $IN --output eval/fast_rate${rate}.jsonl --sample-rate $rate --workers 16 2> eval/fast_rate${rate}.log
done
$PY run.py --input $IN --output eval/fast_all_fp16.jsonl --sample-rate 0 --workers 16 2> eval/fast_all_fp16.log
echo SWEEP_DONE
