@echo off
REM ClaudeRLM bootstrap — downloads the binary on first run.
REM Called from memory-rlm-serve.cmd; all output goes to stderr
REM so it won't interfere with MCP JSON on stdout.

set "PLUGIN_ROOT=%~dp0.."
set "BINARY=%PLUGIN_ROOT%\bin\memory-rlm.exe"
set "REPO=dullfig/memory-rlm"

REM Fast path: binary already exists
if exist "%BINARY%" exit /b 0

echo [memory-rlm] First run — downloading memory-rlm binary... >&2

REM Get latest release tag
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'https://api.github.com/repos/%REPO%/releases/latest').tag_name"') do set "VERSION=%%i"

if "%VERSION%"=="" (
    echo [memory-rlm] ERROR: Could not determine latest release version. >&2
    echo [memory-rlm] Check https://github.com/%REPO%/releases and download manually. >&2
    exit /b 1
)

set "TARGET=x86_64-pc-windows-msvc"
set "URL=https://github.com/%REPO%/releases/download/%VERSION%/memory-rlm-%TARGET%.zip"
set "TMPDIR=%TEMP%\memory-rlm-bootstrap"

echo [memory-rlm] Downloading %VERSION% for %TARGET%... >&2

if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
mkdir "%TMPDIR%"

powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%TMPDIR%\archive.zip' -UseBasicParsing"
if errorlevel 1 (
    echo [memory-rlm] ERROR: Download failed. >&2
    exit /b 1
)

powershell -NoProfile -Command "Expand-Archive -Path '%TMPDIR%\archive.zip' -DestinationPath '%TMPDIR%' -Force"

if not exist "%PLUGIN_ROOT%\bin" mkdir "%PLUGIN_ROOT%\bin"
move /y "%TMPDIR%\memory-rlm.exe" "%BINARY%" >nul

rmdir /s /q "%TMPDIR%"

echo %VERSION%> "%PLUGIN_ROOT%\bin\.version"

echo [memory-rlm] Installed memory-rlm %VERSION% >&2
