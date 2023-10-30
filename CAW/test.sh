#!/bin/bash

data=$@
n_runs=5
prefix="CAW"
neg_samples="hist_nre induc_nre"

echo "****************************************************************************************************************"
echo "*** Running CAW-N testing ***"
echo "dataset: $@"
echo "n_runs: $n_runs"
echo "prefix: $prefix"
echo "Start Time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "****************************************************************************************************************"

for N in $neg_samples; do
    for D in $data; do
    start_time=$(date +'%H:%M:%S')
    python caw_test_trained_model_main.py -d $D --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --pos_dim 172 --walk_pool attn --seed 123 --n_runs $n_runs --gpu 1 --prefix $prefix --neg_sample $N
        for p in linear root geometric; do
            echo "prefix: ${prefix}_cl_$p"
            python caw_test_trained_model_main.py -d $D --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --pos_dim 172 --walk_pool attn --seed 123 --n_runs $n_runs --gpu 1 --prefix ${prefix}_cl_$p --CL --pacing $p --neg_sample $N
        done
    done
done
end_time=$(date +'%H:%M:%S')

echo "****************************************************************************************************************"
echo "Total elapsed time:"
echo "Start Time: $start_time"
echo "End Time: $end_time"
echo "****************************************************************************************************************"
