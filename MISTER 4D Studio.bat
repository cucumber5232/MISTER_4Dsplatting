@echo off
rem MISTER 4D Studio: the guided interface, for whoever does not use a terminal.
rem Run scripts\install.bat once before the first use.
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
".venv\Scripts\python.exe" -m mister4d.app %*
if errorlevel 1 (
  echo.
  echo MISTER 4D Studio si e' chiuso con un errore: il messaggio qui sopra dice il motivo.
  pause
)
