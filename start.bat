@echo off
chcp 65001 >nul
title VoxCPM2 Voice Cloner - 啟動中...
cd /d "%~dp0"

:: ========== 檢查是否已安裝 ==========
:: 環境在本機、不在專案裡（見 install.ps1）：優先用 media-short-film 的 %AI_ROOT%\Voice\venv，
:: 沒有才用本專案自己的 ~\.local\share\media-voice-clone\.venv
if not defined AI_ROOT set "AI_ROOT=C:\AI"
set "PY=%AI_ROOT%\Voice\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\.local\share\media-voice-clone\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ========================================
    echo   尚未安裝，請先雙擊 install.bat
    echo ========================================
    echo.
    pause
    exit /b 1
)

:: 伺服器負責開啟瀏覽器、辨識同一專案的既有服務並回報連接埠衝突。
"%PY%" record_ui.py --open
if errorlevel 1 (
    pause
    exit /b 1
)
