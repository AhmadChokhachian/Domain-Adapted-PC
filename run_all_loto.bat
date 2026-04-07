@echo off
cd /d "%~dp0"
set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo ============================================
echo Running ALL LOTO scripts
echo Output directory: %OUTDIR%
echo ============================================
echo.

:: ---- matching ----
echo [1/12] Running matching.py ...
python code\matching.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table1 LOTO ----
echo [2/12] Running Table1.R weighted_ks 2017 ...
Rscript --vanilla code\Table1.R weighted_ks 2017
if errorlevel 1 goto :error
echo [3/12] Running Table1.R weighted_ks 2018 ...
Rscript --vanilla code\Table1.R weighted_ks 2018
if errorlevel 1 goto :error
echo [4/12] Running Table1.R mean_ks 2017 ...
Rscript --vanilla code\Table1.R mean_ks 2017
if errorlevel 1 goto :error
echo [5/12] Running Table1.R mean_ks 2018 ...
Rscript --vanilla code\Table1.R mean_ks 2018
if errorlevel 1 goto :error
echo [6/12] Running Table1.R marginal_energy 2017 ...
Rscript --vanilla code\Table1.R marginal_energy 2017
if errorlevel 1 goto :error
echo [7/12] Running Table1.R marginal_energy 2018 ...
Rscript --vanilla code\Table1.R marginal_energy 2018
if errorlevel 1 goto :error
echo [8/12] Running Table1.R sinkhorn_wasserstein 2017 ...
Rscript --vanilla code\Table1.R sinkhorn_wasserstein 2017
if errorlevel 1 goto :error
echo [9/12] Running Table1.R sinkhorn_wasserstein 2018 ...
Rscript --vanilla code\Table1.R sinkhorn_wasserstein 2018
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 TF ----
echo [10/12] Running Table2_TF_thinned_SV.R 2017 ...
Rscript --vanilla "code\Table2_TF_thinned_SV.R" 2017
if errorlevel 1 goto :error
echo Running Table2_TF_thinned_SV.R 2018 ...
Rscript --vanilla "code\Table2_TF_thinned_SV.R" 2018
if errorlevel 1 goto :error
echo Running Table2_TF_ANN.py 2017 ...
python code\Table2_TF_ANN.py 2017
if errorlevel 1 goto :error
echo Running Table2_TF_ANN.py 2018 ...
python code\Table2_TF_ANN.py 2018
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 G ----
echo [11/12] Running Table2_G scripts ...
Rscript --vanilla code\Table2_G_rf.R 2017
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_rf.R 2018
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_SVR.R 2017
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_SVR.R 2018
if errorlevel 1 goto :error
python code\Table2_G_XGBoost.py 2017
if errorlevel 1 goto :error
python code\Table2_G_XGBoost.py 2018
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 P ----
echo [12/12] Running Table2_P scripts ...
Rscript --vanilla code\Table2_P_binning.R
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_twinGP.R 2017
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_twinGP.R 2018
if errorlevel 1 goto :error
python code\Table2_P_XGBoost.py 2017
if errorlevel 1 goto :error
python code\Table2_P_XGBoost.py 2018
if errorlevel 1 goto :error
python code\Table2_P_GNN.py 2017
if errorlevel 1 goto :error
python code\Table2_P_GNN.py 2018
if errorlevel 1 goto :error
echo Done.
echo.

echo ============================================
echo ALL LOTO scripts finished successfully.
echo ============================================
pause
exit /b 0

:error
echo.
echo FAILED. Check output above.
pause
exit /b 1
