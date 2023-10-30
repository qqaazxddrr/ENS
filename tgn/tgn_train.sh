#!/bin/bash

n_runs=5

for d in $1; do
  for m in jodie dyrep tgn; do
    if [ "$m" == "tgn" ]; then
      prefix=tgn_attn
    else
      prefix=${m}_rnn
    fi

    echo "****************************************************************************************************************"
    echo "dataset: $d"
    echo "prefix: $prefix"
    echo "n_runs: $n_runs"
    echo "Start Time: $(date)"
    echo "****************************************************************************************************************"

    start_time=$(date +%s)

    if [ "$m" == "tgn" ]; then
      python train_self_supervised.py -d $d --use_memory --prefix $prefix --n_runs $n_runs --gpu 0
    elif [ "$m" == "jodie" ]; then
      python train_self_supervised.py -d $d --use_memory --memory_updater rnn --embedding_module time --prefix $prefix --n_runs $n_runs --gpu 0
    elif [ "$m" == "dyrep" ]; then
      python train_self_supervised.py -d $d --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix $prefix --n_runs $n_runs --gpu 0
    elif [ "$m" == "preproc" ]; then
      python utils/preprocess_data.py --data $d
    else
      echo "Undefined task!"
    fi

    end_time=$(date +%s)

    echo "****************************************************************************************************************"
    echo "Method: $m, Data: $d: Elapsed Time:"
    echo "Start Time: $start_time"
    echo "End Time: $end_time"
    echo "****************************************************************************************************************"
    echo
    echo
  done
done
