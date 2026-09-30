@echo off
cd /d "%~dp0"
title VoiceLink sound check
echo ==============================================================
echo   VoiceLink sound check
echo   Make VoiceLink speak now (type a message in the page).
echo   For 20 seconds this unmutes Chrome / Edge in the Windows
echo   Volume Mixer and then shows every app's playback level.
echo ==============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File tools\audio_sessions.ps1 -Unmute chrome,msedge -Seconds 20
powershell -NoProfile -ExecutionPolicy Bypass -File tools\audio_sessions.ps1 -Seconds 5
echo   Mute = True        : that app is muted in the Volume Mixer.
echo   MaxPeak above 0    : the app is producing sound on that device.
echo   Check that the [default] device is the speaker/headset you listen to.
if not defined VOICELINK_NOPAUSE pause
