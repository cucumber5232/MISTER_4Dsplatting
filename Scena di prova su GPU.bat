@echo off
rem The test scene, whole and trained long, on the GPUs of this computer.
rem Double click, then leave the window open: it takes a few hours.
rem At the end: consegna\ holds the player and the video, risultati.zip is the report to send back.
setlocal
cd /d "%~dp0"
title MISTER 4D - scena di prova su GPU
rem All that follows is one block: it is read whole before it starts, so the update below can
rem rewrite this file while it runs.
(
  if exist ".git" (
    where git >nul 2>nul && (
      echo Cerco aggiornamenti del programma...
      git pull --ff-only || echo Aggiornamento non riuscito: continuo con la versione che c'e' gia'.
      echo.
    )
  )
  if not exist ".venv\Scripts\python.exe" (
    echo MISTER 4D non e' ancora installato su questo computer: lo installo adesso.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
  )
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Installazione non riuscita: il messaggio qui sopra dice il motivo.
    pause
    exit /b 1
  )
  set "PYTHONPATH=%~dp0src"
  ".venv\Scripts\python.exe" scripts\scena_di_prova.py %*
  if errorlevel 1 (
    echo.
    echo Qualcosa non e' andato a buon fine. Manda comunque il file risultati.zip: contiene il motivo.
  ) else (
    echo.
    echo Finito. Player e video sono nella cartella consegna. Il file da mandare e' risultati.zip.
  )
  pause
  exit /b
)
