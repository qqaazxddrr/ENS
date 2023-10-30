@echo off
setlocal enabledelayedexpansion

:loop
if "%~1"=="" goto end

set dataset=%~1
call tgn_train.bat %dataset%
call tgn_train_cl.bat %dataset%
call tgn_test.bat %dataset%
call tgn_test_cl.bat %dataset%

shift
goto loop

:end
