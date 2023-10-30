@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM commands
set "n_runs=5"
set "methods=jodie dyrep tgn"
set "neg_samples=hist_nre induc_nre"
set "pacing=linear root geometric"

FOR %%D IN (%1) DO (
  FOR %%M IN (%methods%) DO (
    FOR %%N IN (%neg_samples%) DO (
      FOR %%P IN (%pacing%) DO (
        echo ****************************************************************************************************************
        echo dataset: %%D
        echo method: %%M
        echo neg_sample: %%N
        echo n_runs: !n_runs!
        echo pacing: %%P
        echo Start Time: %date% %time%
        echo ****************************************************************************************************************

        SET start_time=%time%
        
        if "%%M"=="tgn" (
          python tgn_test_trained_model_self_sup.py -d %%D --use_memory --model %%M --gpu 0 --neg_sample %%N --n_runs !n_runs! --cl --pacing %%P
        ) else if "%%M"=="jodie" (
          python tgn_test_trained_model_self_sup.py -d %%D --use_memory --memory_updater rnn --embedding_module time --model %%M --gpu 0 --neg_sample %%N --n_runs !n_runs! --cl --pacing %%P
        ) else if "%%M"=="dyrep" (
          python tgn_test_trained_model_self_sup.py -d %%D --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --model %%M --gpu 0 --neg_sample %%N --n_runs !n_runs! --cl --pacing %%P
        ) else (
          echo Undefined task!
        )

        SET end_time=%time%
        echo *******************************************************
        echo Method: %%M, NEG_SAMPLE: %%N, Data: %%D,, PACINGL %%P, Elapsed Time: !start_time! to !end_time!.
        echo ****************************************************************************************************************
        echo.
      )
    )
  )
)



