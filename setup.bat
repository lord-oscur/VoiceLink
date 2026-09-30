@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title VoiceLink setup

echo ==============================================================
echo   VoiceLink setup  -  real-time voice chat for LM Studio
echo ==============================================================
echo   Usage:  setup.bat        (CPU speech engines - recommended)
echo           setup.bat gpu    (CUDA build of ONNX Runtime, NVIDIA only)
echo           setup.bat cpu    (switch back to the CPU build)
echo.

set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :have_venv

rem ---- find a Python 3.10 - 3.13 (3.12 preferred) ------------------------
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python 3.10 - 3.13 was not found.
    echo         Install Python 3.12 from https://www.python.org/downloads/ and run setup.bat again.
    goto :fail
)
echo [1/4] Creating virtual environment with: !PY!
!PY! -m venv .venv || goto :fail

:have_venv
echo [2/4] Installing Python packages...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check || goto :fail

rem ---- CPU or GPU build of ONNX Runtime (remembered between runs) -------
set "MODE=cpu"
if exist ".venv\gpu.flag" set "MODE=gpu"
if /i "%~1"=="gpu" set "MODE=gpu"
if /i "%~1"=="cpu" set "MODE=cpu"

echo [3/4] ONNX Runtime build: !MODE!
if /i "!MODE!"=="gpu" (
    "%VPY%" -m pip uninstall -y onnxruntime >nul 2>&1
    "%VPY%" -m pip install -r requirements-gpu.txt --disable-pip-version-check || goto :fail
    "%VPY%" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.26.*" --disable-pip-version-check --quiet || goto :fail
    echo gpu> ".venv\gpu.flag"
    echo       GPU build installed. Set "device": "cuda" for "stt" and/or "tts" in config.json to use it.
) else (
    "%VPY%" -m pip show onnxruntime-gpu >nul 2>&1 && (
        "%VPY%" -m pip uninstall -y onnxruntime-gpu >nul 2>&1
        "%VPY%" -m pip install --force-reinstall --no-deps "onnxruntime>=1.20,<2" --disable-pip-version-check --quiet || goto :fail
    )
    if exist ".venv\gpu.flag" del ".venv\gpu.flag"
)

echo [4/4] Downloading models (about 1 GB, first time only)...
"%VPY%" tools\download_models.py || goto :fail

echo.
echo ==============================================================
echo   Setup complete.
echo   1. In LM Studio: load a model and start the server
echo      (Developer tab, default http://127.0.0.1:1234).
echo   2. Double-click run.bat - the voice chat opens in your browser.
echo ==============================================================
if not defined VOICELINK_NOPAUSE pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed - see the messages above.
if not defined VOICELINK_NOPAUSE pause
exit /b 1
