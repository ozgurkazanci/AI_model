@echo off
REM Double-click launcher for mikroelektronix.
REM
REM It exists because the app needs two things set that a double-click does not
REM provide: the working directory must be the repository root (the model path
REM and configs/local_inference.yaml are resolved from there) and PYTHONPATH
REM must include src. Running "python app.py" from Explorer gets neither and
REM fails with an import error that says nothing useful.
REM
REM Pass --serve to start the model with the window:  mikroelektronix.bat --serve

setlocal

REM This file lives in mikroelektronix\, so the repository root is one level up.
set "REPO=%~dp0.."
cd /d "%REPO%" || (echo Could not enter %REPO% & pause & exit /b 1)

set "PYTHONPATH=src"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python 3.11 or add it to PATH, then run this again.
  pause
  exit /b 1
)

python -c "import webview" >nul 2>&1
if errorlevel 1 (
  echo pywebview is not installed. Installing it now...
  python -m pip install pywebview || (echo Install failed. & pause & exit /b 1)
)

echo Starting mikroelektronix...
REM A double-click passes no arguments, and then the window would open with
REM nothing behind it. Default to --serve; the app reuses a server that is
REM already running rather than starting a second one.
if "%~1"=="" (
  python -m mikroelektronix.app --serve
) else (
  python -m mikroelektronix.app %*
)

REM Only pause when something went wrong, so a normal close does not leave a
REM console window sitting there.
if errorlevel 1 pause
endlocal
