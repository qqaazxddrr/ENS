#!/bin/bash

#####################################
# parameters & methods
#####################################
prefix="TGAT"
n_runs=5

#####################################
# commands
#####################################

for data in $@; do
  for neg_sample in hist_nre induc_nre; do
    start_time="$(date -u +%s)"
    echo "****************************************************************************************************************"
    echo "*** Running tgat_run.sh: TGAT method execution ***"
    echo "dataset: $data"
    echo "prefix: $prefix"
    echo "neg_sample: $neg_sample"
    echo "n_runs: $n_runs"
    echo "Start Time: $(date)"
    echo "****************************************************************************************************************"

    python tgat_test_trained_model_learn_edge.py -d $data --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs --neg_sample $neg_sample

    for p in linear root geometric; do
      python tgat_test_trained_model_learn_edge.py -d $data --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix "${prefix}_cl_$p" --n_runs $n_runs --neg_sample $neg_sample
    done

    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    elapsed_formatted=$(date -u -d @${elapsed} +"%T")
    echo "******************************************************"
    echo "Method: $prefix, Data: $data: Elapsed Time: $elapsed_formatted."
    echo "****************************************************************************************************************"
    echo ""
    echo ""
  done
done
