@echo off
setlocal enabledelayedexpansion

rem === Configuration ===
rem Pass the root folder as the first argument, or default to current folder
set "ROOT_FOLDER=%~1"
if "%ROOT_FOLDER%"=="" set "ROOT_FOLDER=%~dp0"

echo ======================================================
echo Running all Python scripts in folder:
echo %ROOT_FOLDER%
echo ======================================================

rem Change directory to the target folder
cd /d "%ROOT_FOLDER%"

rem Loop through all .py files in the folder (non-recursive)
for %%F in (*.py) do (
    echo ------------------------------------------------------
    echo Running: %%F
    echo ------------------------------------------------------
    python "%%F"
    echo.
)

echo ======================================================
echo ✅ All Python scripts executed successfully.
echo ======================================================

pause
endlocal
