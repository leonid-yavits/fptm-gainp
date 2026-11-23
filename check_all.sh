#!/bin/sh
# Bit-exact checks (Mapping S: 1,500 test samples, Mapping C: 200) for every run without a result.
for d in results/*/*/; do
  ds=$(basename $(dirname $d)); tag=$(basename $d)
  [ -f $d/model.pkl ] || continue
  [ -f $d/bitexact.txt ] && continue
  python3 check_gainp.py $ds $tag 1500 200 > $d/bitexact.txt 2>&1
done
