@echo off
setlocal enabledelayedexpansion

set prefix=TGAT
set n_runs=5

for %%I in (%1) do (
  for %%J in (hist_nre induc_nre) do (
    echo ****************************************************************************************************************
    echo *** Running tgat_run.bat: TGAT method execution ***
    echo dataset: %%I
    echo prefix: !prefix!
    echo neg_sample: %%J
    echo n_runs: !n_runs!
    echo ****************************************************************************************************************


    python tgat_test_trained_model_learn_edge.py -d %%I --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix !prefix! --n_runs !n_runs! --neg_sample %%J

    for %%P in (linear root geometric) do (
      python tgat_test_trained_model_learn_edge.py -d %%I --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix "!prefix!_cl_%%P" --n_runs !n_runs! --neg_sample %%J
    )

    echo ******************************************************
    echo Method: !prefix!, Data: %%I
    echo ****************************************************************************************************************
    echo.
    echo.
  )
)
