@echo off
chcp 65001 >nul
title VoxCPM2 Voice Cloner - 安裝中...
cd /d "%~dp0"

echo ============================================
echo   VoxCPM2 Voice Cloner - 一鍵安裝
echo ============================================
echo.
echo 本安裝需要：
echo   - PowerShell 7（pwsh）
echo   - 約 5GB 硬碟空間（模型權重）；Python 3.12 由 uv 自動安裝
echo   - 麥克風
echo.

:: ========== 檢查 PowerShell 7 ==========
:: install.ps1 需要 pwsh；Windows 內建的 powershell.exe 是 5.1，跑不了
where pwsh >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 PowerShell 7（pwsh），請先安裝：
    echo   winget install --id Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

:: ========== 執行安裝腳本 ==========
pwsh -NoProfile -ExecutionPolicy Bypass -File "install.ps1"

if errorlevel 1 (
    echo.
    echo [錯誤] 安裝失敗，請查看上方錯誤訊息。
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   安裝完成！
echo   關閉此視窗，雙擊 start.bat 即可使用。
echo ============================================
echo.
pause
