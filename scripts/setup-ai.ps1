$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install -e .
Write-Host 'MSAS AI backend is ready. Run npm start.' -ForegroundColor Green
