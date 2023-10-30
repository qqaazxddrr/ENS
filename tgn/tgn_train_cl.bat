@echo off
setlocal enabledelayedexpansion

set n_runs=5

for %%d in (%1) do (
  for %%m in (jodie dyrep tgn) do (
    for %%p in (linear root geometric) do (
      if "%%m"=="tgn" (
        set prefix=tgn_attn_cl_%%p
      ) else (
        set prefix=%%m_rnn_cl_%%p
      )

      echo ****************************************************************************************************************
      echo dataset: %%d
      echo prefix: !prefix!
      echo n_runs: %n_runs%
      echo pacing %P
      echo Start Time: %date% %time%
      echo ****************************************************************************************************************

      set start_time=%time%

      if "%%m"=="tgn" (
        echo ^^^^> train_self_supervised; TGN; data: %%d
        python train_self_supervised.py -d %%d --use_memory --prefix !prefix! --n_runs %n_runs% --gpu 0 --CL --pacing %%p
      ) else if "%%m"=="jodie" (
        echo ^^^^> train_self_supervised; jodie_rnn; data: %%d
        python train_self_supervised.py -d %%d --use_memory --memory_updater rnn --embedding_module time --prefix !prefix! --n_runs %n_runs% --gpu 0 --CL --pacing %%p
      ) else if "%%m"=="dyrep" (
        echo ^^^^> train_self_supervised; dyrep_rnn; data: %%d
        python train_self_supervised.py -d %%d --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix !prefix! --n_runs %n_runs% --gpu 0 --CL --pacing %%p
      ) else if "%%m"=="preproc" (
        echo ^^^^> Preprocessing data!
        python utils/preprocess_data.py --data %%d
      ) else (
        echo Undefined task!
      )

      set end_time=%time%

      echo ****************************************************************************************************************
      echo Method: %%m, Data: %%d: Elapsed Time:
      echo Start Time: !start_time!
      echo End Time: !end_time!
      echo ****************************************************************************************************************
      echo.
      echo.
    )
  )
)