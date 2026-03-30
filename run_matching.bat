@echo off
cd /d "%~dp0"

set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running matching...
echo Output directory: %OUTDIR%
echo.

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set start=%%i

python -m jupyter nbconvert --to notebook --execute "code\matching.ipynb" --inplace
if errorlevel 1 goto :error

for /f %%i in ('powershell -NoProfile -Command "[int][double](Get-Date -UFormat %%s)"') do set end=%%i
set /a runtime=%end%-%start%

echo Matching runtime (sec): %runtime%
>"%OUTDIR%\matching_runtime.txt" echo %runtime%

echo.
echo Matching finished successfully.
pause
exit /b 0

:error
echo.
echo Matching failed.
pause
exit /b 1