@echo off
cd /d "%~dp0"
title VoiceLink
if not exist ".venv\Scripts\python.exe" (
    echo VoiceLink is not installed yet - run setup.bat first.
    pause
    exit /b 1
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
".venv\Scripts\python.exe" -m voicelink
if errorlevel 1 if not defined VOICELINK_NOPAUSE pause
