@echo off
cd /d "%~dp0"

set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running Tables 2-3...
echo Output directory: %OUTDIR%
echo.

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set script_start=%%i

:: =========================
:: STGP 
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2-Table3(STGP).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a stgp_runtime=%end%-%start%
echo STGP runtime (sec): %stgp_runtime%
>"%OUTDIR%\stgp_runtime.txt" echo %stgp_runtime%

:: =========================
:: twinGP + Binning 
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2_Table3(twinGP+Binning).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a twingp_runtime=%end%-%start%
echo twinGP+Binning runtime (sec): %twingp_runtime%
>"%OUTDIR%\twingp_runtime.txt" echo %twingp_runtime%

:: =========================
:: NN 
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2-Table3(NN).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a nn_runtime=%end%-%start%
echo NN runtime (sec): %nn_runtime%
>"%OUTDIR%\nn_runtime.txt" echo %nn_runtime%

:: =========================
:: BNN 
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2-Table3(BNN).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a bnn_runtime=%end%-%start%
echo BNN runtime (sec): %bnn_runtime%
>"%OUTDIR%\bnn_runtime.txt" echo %bnn_runtime%

:: =========================
:: XGBoost
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2-Table3(XGBoost).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a xgb_runtime=%end%-%start%
echo XGBoost runtime (sec): %xgb_runtime%
>"%OUTDIR%\xgboost_runtime.txt" echo %xgb_runtime%

:: =========================
:: Update final results
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python "code\update_final_results.py"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a python_runtime=%end%-%start%

echo Python update runtime (sec): %python_runtime%
>"%OUTDIR%\python_runtime.txt" echo %python_runtime%

:: =========================
:: TOTAL SCRIPT TIME
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set script_end=%%i
set /a total_runtime=%script_end%-%script_start%

echo.
echo Total runtime (sec): %total_runtime%
>"%OUTDIR%\table23_total_runtime.txt" echo %total_runtime%

echo.
echo Files written:
dir "%OUTDIR%"

echo.
echo Tables 2-3 finished successfully.
pause
exit /b 0

:error
echo.
echo Tables 2-3 failed.
echo Existing files in output directory:
dir "%OUTDIR%"
pause
exit /b 1