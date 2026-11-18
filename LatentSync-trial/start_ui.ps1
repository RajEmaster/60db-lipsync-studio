$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.ui-venv\Scripts\python.exe'
& $pythonPath -u (Join-Path $PSScriptRoot 'local_ui.py')
