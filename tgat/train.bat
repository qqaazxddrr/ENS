@echo off

set prefix=TGAT
set mode=self_sup_link
set n_runs=5

for %%d in (USLegis) do (
    echo ****************************************************************************************************************
    echo *** Running tgat_run.bat: TGAT method execution ***
    echo dataset: %%d
    echo prefix: %prefix%
    echo mode: %mode%
    echo n_runs: %n_runs%
    echo Start Time: %date% %time%
    echo ****************************************************************************************************************

    python -u learn_edge.py -d %%d --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix %prefix% --n_runs %n_runs%
    
    for %%p in (linear root geometric) do (
        set prefix=%prefix%_cl_%%p
        echo prefix: %prefix%
        python -u learn_edge.py -d %%d --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix %prefix% --n_runs %n_runs% --CL --pacing %%p
    )

    echo *******************************************************
    echo Method: %prefix%, Data: %%d: End Time: %date% %time%
    echo ****************************************************************************************************************
    echo.
    echo.
)
