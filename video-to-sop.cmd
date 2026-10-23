@echo off
setlocal
set "VTS_ROOT=%~dp0"
set "VTS_PYTHON=%VTS_ROOT%runtime\python\python.exe"
if not exist "%VTS_PYTHON%" (
  echo Portable runtime is missing. Use the portable ZIP, or follow README.md for source installation. 1>&2
  exit /b 3
)
if not exist "%VTS_ROOT%config.local.toml" (
  copy /y "%VTS_ROOT%config.example.toml" "%VTS_ROOT%config.local.toml" >nul
  if errorlevel 1 (
    echo Could not create config.local.toml. Check write permission in the Skill folder. 1>&2
    exit /b 3
  )
)
set "PATH=%VTS_ROOT%runtime\ffmpeg;%PATH%"
set "PLAYWRIGHT_BROWSERS_PATH=%VTS_ROOT%runtime\browsers"
set "PYTHONUTF8=1"
"%VTS_PYTHON%" -m video_to_sop.cli --config "%VTS_ROOT%config.local.toml" %*
set "VTS_EXIT=%ERRORLEVEL%"
exit /b %VTS_EXIT%
