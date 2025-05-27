@echo off
title Platform3 Comprehensive Validation Suite

echo.
echo =============================================================
echo   Platform3 67-Indicator System - Comprehensive Validation
echo =============================================================
echo.

REM Execute the PowerShell validation script
powershell -ExecutionPolicy Bypass -File "%~dp0run-validation.ps1"

echo.
echo Press any key to continue...
pause >nul
