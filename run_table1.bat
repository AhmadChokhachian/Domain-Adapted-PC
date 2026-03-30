@echo off
cd /d "%~dp0"

set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running Table 1...
echo Output directory: %OUTDIR%
echo.

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i

"C:\Users\achokhachian3\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe" "code\Table1.R"
if errorlevel 1 goto :error

if not exist "%OUTDIR%\Table1_summary_all_metrics.csv" goto :error

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a runtime=%end%-%start%

echo Table 1 runtime (sec): %runtime%
>"%OUTDIR%\table1_runtime.txt" echo %runtime%

echo.
echo Table 1 finished successfully.
pause
exit /b 0

:error
echo.
echo Table 1 failed.
echo Existing files in output directory:
dir "%OUTDIR%"
pause
exit /b 1