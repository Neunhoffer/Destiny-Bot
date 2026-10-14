@echo off
rem Athena launcher for Windows: double-click to start.
rem First run: creates a private Python environment in this folder (.venv),
rem installs Athena, downloads a natural offline voice (~60 MB) and puts an
rem Athena icon on the desktop and in the Start menu.
rem All books, settings and audio stay in the "data" folder next to this file.
setlocal
cd /d "%~dp0"
set "DESTINY_HOME=%~dp0data"
set "VENV=%~dp0.venv"

if exist "%VENV%\destiny-installed.txt" goto installed

echo.
echo  Setting up Athena for the first time. This takes a few minutes...
echo.
set "PYCMD="
where py >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD (
  where python >nul 2>nul && set "PYCMD=python"
)
if not defined PYCMD goto nopython
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto nopython

if exist "%VENV%" rmdir /s /q "%VENV%"
%PYCMD% -m venv "%VENV%"
if errorlevel 1 goto failed
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -e ".[ai,edge,ocr,system]"
if errorlevel 1 goto failed
rem The natural offline voices are optional: keep going if this computer can't install them
rem (they need onnxruntime, which can lag behind the newest Python for a while).
"%VENV%\Scripts\python.exe" -m pip install "piper-tts>=1.3"
if errorlevel 1 echo  Note: natural offline voices aren't available on this computer; the system voice will be used.
"%VENV%\Scripts\python.exe" -m pip install -e ".[kokoro]"
if errorlevel 1 echo  Note: the Kokoro voices aren't available on this computer.
rem The app window is optional too: without it Athena opens in a Microsoft Edge window.
"%VENV%\Scripts\python.exe" -m pip install "pywebview>=5.0"
echo installed> "%VENV%\destiny-installed.txt"

:installed
if not exist "%DESTINY_HOME%\voices\piper\*.onnx" (
  echo.
  echo  Downloading a natural-sounding offline voice, about 60 MB, one time only...
  "%VENV%\Scripts\python.exe" -m destiny_bot voices download en_US-lessac-medium
)
if not exist "%VENV%\destiny-shortcut.txt" (
  echo.
  echo  Adding an Athena icon to your desktop and Start menu...
  "%VENV%\Scripts\python.exe" -m destiny_bot shortcut && echo created> "%VENV%\destiny-shortcut.txt"
)
rem pythonw has no console window, so this window can close while Athena runs.
set "PYW=%VENV%\Scripts\pythonw.exe"
if not exist "%PYW%" set "PYW=%VENV%\Scripts\python.exe"
start "" "%PYW%" -m destiny_bot app
goto end

:nopython
echo.
echo  Athena needs Python 3.10 or newer.
echo  1. Download it from https://www.python.org/downloads/
echo  2. In the installer, tick "Add python.exe to PATH".
echo  3. Double-click this file again.
echo.
pause
goto end

:failed
echo.
echo  Setup didn't finish. Check your internet connection and try again.
echo  To start over, delete the ".venv" folder in this folder.
echo.
pause

:end
endlocal
