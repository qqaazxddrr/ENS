#!/bin/bash

prefix=TGAT
mode=self_sup_link
n_runs=5
gpu=$2

for d in $1; do
  echo "****************************************************************************************************************"
  echo "*** Running tgat_run.sh: TGAT method execution ***"
  echo "dataset: $d"
  echo "prefix: $prefix"
  echo "mode: $mode"
  echo "n_runs: $n_runs"
  echo "Start Time: $(date)"
  echo "****************************************************************************************************************"

  python -u learn_edge.py -d $d --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu $gpu --n_head 2 --prefix $prefix --n_runs $n_runs
  
  
  for p in linear root geometric; do
    python -u learn_edge.py -d $d --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu $gpu --n_head 2 --prefix "${prefix}_cl_${p}" --n_runs $n_runs --CL --pacing $p
  done
 
  echo "*******************************************************"
  echo "Method: $prefix, Data: $d: End Time: $(date)"
  echo "****************************************************************************************************************"
  echo
  echo
done

