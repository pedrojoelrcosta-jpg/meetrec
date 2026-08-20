# meetrec one-shot installer for Windows.
#   git clone https://github.com/pedrojoelrcosta-jpg/meetrec
#   cd meetrec
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Creates the venv, installs everything, then launches the interactive
# `meetrec setup` wizard (keys, Telegram pairing, preferences, autostart).

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

Write-Host "`n=== meetrec installer ===`n"

# 1. Python 3.11+
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Python launcher 'py' not found." -ForegroundColor Red
    Write-Host "Install Python 3.11+ from https://www.python.org/downloads/ (check 'Add to PATH'), then re-run."
    exit 1
}
$version = & py -3 -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]'3.11') {
    Write-Host "Python $version found, but 3.11+ is required." -ForegroundColor Red
    exit 1
}
Write-Host "[ok] Python $version"

# 2. venv — use a short path when the clone is deep (torch hits MAX_PATH ~260)
$venv = Join-Path $root '.venv'
if ($root.Length -gt 60) {
    $venv = 'C:\venvs\meetrec'
    Write-Host "[i]  Repo path is long; using $venv to avoid Windows MAX_PATH issues"
}
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    Write-Host "[..] Creating venv at $venv"
    & py -3 -m venv $venv
}
$python = Join-Path $venv 'Scripts\python.exe'
Write-Host "[ok] venv ready"

# 3. dependencies (torch + pyannote + faster-whisper: several GB on first run)
Write-Host "[..] Installing dependencies (first run downloads a few GB - be patient)"
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -e $root
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed - see output above." -ForegroundColor Red
    exit 1
}
Write-Host "[ok] Dependencies installed"

# 4. interactive setup wizard (keys, Telegram, preferences, autostart)
& $python -m meetrec setup

Write-Host "`nInstalled. Useful commands:"
Write-Host "  $python -m meetrec doctor"
Write-Host "  $python -m meetrec start"
Write-Host "`nTip: add $(Join-Path $venv 'Scripts') to PATH to use 'meetrec' directly."
