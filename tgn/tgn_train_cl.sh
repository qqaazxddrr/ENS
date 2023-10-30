#!/bin/bash

n_runs=5

for d in $1; do
  for m in jodie dyrep tgn; do
    for p in linear root geometric; do
      if [ "$m" == "tgn" ]; then
        prefix="tgn_attn_cl_$p"
      else
        prefix="${m}_rnn_cl_$p"
      fi

      echo "****************************************************************************************************************"
      echo "dataset: $d"
      echo "prefix: $prefix"
      echo "n_runs: $n_runs"
      echo "pacing: $p"
      echo "Start Time: $(date +"%Y-%m-%d %H:%M:%S")"
      echo "****************************************************************************************************************"

      start_time=$(date +"%Y-%m-%d %H:%M:%S")

      if [ "$m" == "tgn" ]; then
        echo "> train_self_supervised; TGN; data: $d"
        python train_self_supervised.py -d $d --use_memory --prefix $prefix --n_runs $n_runs --gpu 0 --CL --pacing $p
      elif [ "$m" == "jodie" ]; then
        echo "> train_self_supervised; jodie_rnn; data: $d"
        python train_self_supervised.py -d $d --use_memory --memory_updater rnn --embedding_module time --prefix $prefix --n_runs $n_runs --gpu 0 --CL --pacing $p
      elif [ "$m" == "dyrep" ]; then
        echo "> train_self_supervised; dyrep_rnn; data: $d"
        python train_self_supervised.py -d $d --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix $prefix --n_runs $n_runs --gpu 0 --CL --pacing $p
      elif [ "$m" == "preproc" ]; then
        echo "> Preprocessing data!"
        python utils/preprocess_data.py --data $d
      else
        echo "Undefined task!"
      fi

      end_time=$(date +"%Y-%m-%d %H:%M:%S")

      echo "****************************************************************************************************************"
      echo "Method: $m, Data: $d: Elapsed Time:"
      echo "Start Time: $start_time"
      echo "End Time: $end_time"
      echo "****************************************************************************************************************"
      echo
      echo
    done
  done
done
