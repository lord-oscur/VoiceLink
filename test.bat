@echo off
rem End-to-end self-test: starts a server (unless one is already running),
rem then talks to it with synthetic speech. No microphone needed.
cd /d "%~dp0"
title VoiceLink self-test
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
rem A separate port keeps an open browser tab from joining the test.
set VOICELINK_PORT=8799
start "VoiceLink test server" /min cmd /c "set VOICELINK_NO_BROWSER=1&& .venv\Scripts\python.exe -m voicelink"
".venv\Scripts\python.exe" tools\selftest.py %*
taskkill /fi "WINDOWTITLE eq VoiceLink test server*" /t /f >nul 2>&1
if not defined VOICELINK_NOPAUSE pause
