#!/bin/sh

n_runs=5
methods="jodie dyrep tgn"
neg_samples="hist_nre induc_nre"
datasets=$1
pacing="linear root geometric"

for D in $datasets; do
  for M in $methods; do
    for N in $neg_samples; do
      for P in $pacing; do
        echo "****************************************************************************************************************"
        echo "dataset: $D"
        echo "method: $M"
        echo "neg_sample: $N"
        echo "n_runs: $n_runs"
        echo "pacing: $P"
        echo "Start Time: $(date)"
        echo "****************************************************************************************************************"

        start_time=$(date +%s)

        if [ "$M" = "tgn" ]; then
          python tgn_test_trained_model_self_sup.py -d $D --use_memory --model $M --gpu 0 --neg_sample $N --n_runs $n_runs --cl --pacing $P
        elif [ "$M" = "jodie" ]; then
          python tgn_test_trained_model_self_sup.py -d $D --use_memory --memory_updater rnn --embedding_module time --model $M --gpu 0 --neg_sample $N --n_runs $n_runs --cl --pacing $P
        elif [ "$M" = "dyrep" ]; then
          python tgn_test_trained_model_self_sup.py -d $D --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --model $M --gpu 0 --neg_sample $N --n_runs $n_runs --cl --pacing $P
        else
          echo "Undefined task!"
        fi

        end_time=$(date +%s)

        echo "*******************************************************"
        echo "Method: $M, NEG_SAMPLE: $N, Data: $D, PACING: $P, Elapsed Time: $start_time to $end_time."
        echo "****************************************************************************************************************"
        echo
      done
    done
  done
done




