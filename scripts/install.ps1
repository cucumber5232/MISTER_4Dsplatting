<#
Windows: create the virtual environment, install PyTorch (CUDA build), the
pre-compiled gsplat wheel and the project, then check the machine.

Run it from the repository folder:

    powershell -ExecutionPolicy Bypass -File scripts\install.ps1

or double-click scripts\install.bat.

Why these exact versions: the official pre-compiled gsplat wheels for Windows
exist only for Python 3.10 with PyTorch 2.4 and CUDA 12.4. With them nothing
has to be compiled, so neither Visual Studio nor the CUDA Toolkit is needed:
the NVIDIA driver is enough.

Requirements: Python 3.10 (python.org installer), a recent NVIDIA driver,
FFmpeg in PATH (for example: winget install Gyan.FFmpeg).
#>
param(
    # Python 3.10 interpreter to use; by default the "py" launcher is asked for 3.10.
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$TorchVersion = "2.4.1"
$TorchIndex = "https://download.pytorch.org/whl/cu124"
$GsplatVersion = "1.5.3"
$GsplatIndex = "https://docs.gsplat.studio/whl/pt24cu124"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

# --- Python 3.10 -------------------------------------------------------------
$PyExe = "python"
$PyArgs = @()
if ($Python -ne "") {
    $PyExe = $Python
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PyExe = "py"
    $PyArgs = @("-3.10")
}
$Found = ""
try { $Found = (& $PyExe @PyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])") } catch { }
if ("$Found".Trim() -ne "3.10") {
    Write-Host "Python 3.10 is required (found: '$Found')." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/windows/ and run this script again,"
    Write-Host "or pass its path:  scripts\install.ps1 -Python C:\path\to\python.exe"
    exit 1
}

# --- virtual environment and packages ----------------------------------------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-Checked $PyExe ($PyArgs + @("-m", "venv", ".venv"))
}
$Venv = Join-Path (Get-Location) ".venv\Scripts\python.exe"

Invoke-Checked $Venv @("-m", "pip", "install", "--upgrade", "pip")
# PyTorch built for CUDA 12.4 (the plain "pip install torch" on Windows has no GPU support).
Invoke-Checked $Venv @("-m", "pip", "install", "torch==$TorchVersion", "--index-url", $TorchIndex)
# What gsplat needs, from PyPI; NumPy 1.x is the safe pairing for this PyTorch build.
Invoke-Checked $Venv @("-m", "pip", "install", "numpy<2", "ninja", "jaxtyping", "rich")
# gsplat with its CUDA code already compiled.
Invoke-Checked $Venv @("-m", "pip", "install", "gsplat==$GsplatVersion", "--index-url", $GsplatIndex)
# This project and the remaining dependencies (NumPy stays on 1.x).
Invoke-Checked $Venv @("-m", "pip", "install", "-e", ".[dev]", "numpy<2")

# --- checks ------------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "FFmpeg was not found in PATH. Install it (winget install Gyan.FFmpeg) and reopen the terminal."
}
& $Venv -m mister4d.cli doctor

Write-Host ""
Write-Host "Installed."
Write-Host "Guided interface: double-click 'MISTER 4D Studio.bat' in this folder."
Write-Host "Command line: in every new terminal, from this folder, run:"
Write-Host "    .\.venv\Scripts\Activate.ps1        (PowerShell)"
Write-Host "    .venv\Scripts\activate.bat          (cmd)"
Write-Host "then use the 'mister4d' command, for example:  mister4d demo --out sessions\demo --run"
Write-Host "Without activating:  .venv\Scripts\mister4d.exe demo --out sessions\demo --run"
