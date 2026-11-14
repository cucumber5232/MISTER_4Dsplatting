@echo off
rem MISTER 4D Studio: the guided interface, for whoever does not use a terminal.
rem Run scripts\install.bat once before the first use.
rem
rem This file only chooses which program runs; everything else is done by the
rem program itself (src\mister4d\launch.py), so that it never has to be changed
rem when the program is:
rem  - PYTHONPATH makes Python load the code of this very folder, whatever copy is
rem    installed: the page and the engine behind it are always the same version;
rem  - the program installs the libraries an update may ask for, and if the Studio
rem    is already open it shows that one instead of starting a second copy.
setlocal
cd /d "%~dp0"
title MISTER 4D Studio
if not exist ".venv\Scripts\python.exe" (
  echo MISTER 4D Studio non risulta installato su questo computer.
  echo Fai doppio clic su scripts\install.bat, attendi la fine e riapri questo file.
  echo.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m mister4d.launch %*
if errorlevel 1 (
  echo.
  echo MISTER 4D Studio si e' chiuso con un errore: il messaggio qui sopra dice il motivo.
  pause
)
