# Builds the Athena installer for Windows: dist\Athena-Setup-<version>.exe
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-NoKokoro] [-SkipInstaller]
#
# Needs Python 3.10+ (3.12 recommended) and Inno Setup 6 (https://jrsoftware.org/isdl.php,
# or: winget install JRSoftware.InnoSetup). Uses a private build environment in
# packaging\build\venv, downloads the bundled voices (Piper and Kokoro), builds with PyInstaller, runs the
# built app's self-test, then compiles the installer. See packaging\README.md.
param(
    [switch]$NoKokoro,
    [switch]$SkipInstaller,
    [string]$Python = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $Python = 'py -3.12' } else { $Python = 'python' }
}
$venv = Join-Path $root 'packaging\build\venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host "Creating the build environment in $venv"
    Invoke-Expression "$Python -m venv `"$venv`""
    if ($LASTEXITCODE -ne 0) { throw "Couldn't create a Python virtual environment with '$Python'." }
}
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $py -m pip install -e ".[build]"
if ($LASTEXITCODE -ne 0) { throw 'Installing the build requirements failed' }

$buildArgs = @()
if ($NoKokoro) { $buildArgs += '--no-kokoro' }
if ($SkipInstaller) { $buildArgs += '--skip-installer' }
& $py packaging\build.py @buildArgs
if ($LASTEXITCODE -ne 0) { throw "The build failed (exit code $LASTEXITCODE)" }
