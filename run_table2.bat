@echo off
cd /d "%~dp0"

set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running Table 2...
echo Output directory: %OUTDIR%
echo.

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set script_start=%%i

:: =========================
:: TF_thinnedSV
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2(TF_thinned_SV).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a tf_thinnedsv_runtime=%end%-%start%
echo TF_thinnedSV runtime (sec): %tf_thinnedsv_runtime%
>"%OUTDIR%\tf_thinnedsv_runtime.txt" echo %tf_thinnedsv_runtime%

:: =========================
:: TF_ANN
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2(TF_ANN).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a tf_ann_runtime=%end%-%start%
echo TF_ANN runtime (sec): %tf_ann_runtime%
>"%OUTDIR%\tf_ann_runtime.txt" echo %tf_ann_runtime%

:: =========================
:: G_random_forest
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2(G_random_forest).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a g_rf_runtime=%end%-%start%
echo G_random_forest runtime (sec): %g_rf_runtime%
>"%OUTDIR%\g_random_forest_runtime.txt" echo %g_rf_runtime%

:: =========================
:: G_XGBoost
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2(G_XGBoost).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a g_xgb_runtime=%end%-%start%
echo G_XGBoost runtime (sec): %g_xgb_runtime%
>"%OUTDIR%\g_xgboost_runtime.txt" echo %g_xgb_runtime%

:: =========================
:: G_SVR
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2(G_SVR).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a g_svr_runtime=%end%-%start%
echo G_SVR runtime (sec): %g_svr_runtime%
>"%OUTDIR%\g_svr_runtime.txt" echo %g_svr_runtime%

:: =========================
:: P_XGBoost
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2(P_XGBoost).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a p_xgb_runtime=%end%-%start%
echo P_XGBoost runtime (sec): %p_xgb_runtime%
>"%OUTDIR%\p_xgboost_runtime.txt" echo %p_xgb_runtime%

:: =========================
:: P_GNN
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
python -m jupyter nbconvert --to notebook --execute "code\Table2(P_GNN).ipynb" --inplace
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a p_gnn_runtime=%end%-%start%
echo P_GNN runtime (sec): %p_gnn_runtime%
>"%OUTDIR%\p_gnn_runtime.txt" echo %p_gnn_runtime%

:: =========================
:: P_twinGP
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2(P_twinGP).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a p_twingp_runtime=%end%-%start%
echo P_twinGP runtime (sec): %p_twingp_runtime%
>"%OUTDIR%\p_twingp_runtime.txt" echo %p_twingp_runtime%

:: =========================
:: P_Binning
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i
"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table2(P_Binning).R"
if errorlevel 1 goto :error
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a p_binning_runtime=%end%-%start%
echo P_Binning runtime (sec): %p_binning_runtime%
>"%OUTDIR%\p_binning_runtime.txt" echo %p_binning_runtime%

:: =========================
:: TOTAL SCRIPT TIME
:: =========================
for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set script_end=%%i
set /a total_runtime=%script_end%-%script_start%

echo.
echo Total runtime (sec): %total_runtime%
>"%OUTDIR%\table2_total_runtime.txt" echo %total_runtime%

echo.
echo Files written:
dir "%OUTDIR%"

echo.
echo Table 2 finished successfully.
pause
exit /b 0

:error
echo.
echo Table 2 failed.
echo Existing files in output directory:
dir "%OUTDIR%"
pause
exit /b 1