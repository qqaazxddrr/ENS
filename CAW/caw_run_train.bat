@echo off

set data=USLegis
set n_runs=5
set prefix=CAW

echo ****************************************************************************************************************
echo *** Running CAW-N method ***
echo dataset: %data%
echo n_runs: %n_runs%
echo prefix: %prefix%
echo Start Time: %date% %time%
echo ****************************************************************************************************************

set start_time=%time%
python main.py -d %data% --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --pos_dim 172 --walk_pool attn --seed 123 --n_runs %n_runs% --gpu 0 --prefix %prefix%

for %%p in (linear root geometric) do (
        echo prefix: %prefix%_cl_%%p
        python main.py -d %data% --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --pos_dim 172 --walk_pool attn --seed 123 --n_runs %n_runs% --gpu 0 --prefix %prefix%_cl_%%p --CL --pacing %%p
    )


set end_time=%time%

echo ****************************************************************************************************************
echo Total elapsed time: 
echo Start Time: %start_time%
echo End Time: %end_time%
echo ****************************************************************************************************************
