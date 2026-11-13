$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$cpuPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$versionPath = Join-Path $workspaceRoot '.venv-gpu\Lib\site-packages\torch\version.py'
$expectedVersionHash = if (Test-Path -LiteralPath $versionPath) { (Get-FileHash -LiteralPath $versionPath -Algorithm SHA256).Hash } else { $null }
Write-Host 'Optional reinstall: approximately 2.7 GiB download and 4.3 GiB installed. Existing CPU queues are unchanged.'
& $cpuPython -m venv (Join-Path $workspaceRoot '.venv-gpu')
if ($LASTEXITCODE -ne 0) { throw 'GPU venv creation failed' }
$gpuPython = Join-Path $workspaceRoot '.venv-gpu\Scripts\python.exe'
& $gpuPython -m pip install -r (Join-Path $workspaceRoot 'requirements-gnn-cuda.txt')
if ($LASTEXITCODE -ne 0) { throw 'Pinned CUDA torch installation failed' }
$shared = Join-Path $workspaceRoot '.venv-gpu\Lib\site-packages\eda_shared_dependencies.pth'
[IO.File]::WriteAllText($shared, ((Join-Path $workspaceRoot '.venv\Lib\site-packages') + "`n"), [Text.UTF8Encoding]::new($false))
if ($expectedVersionHash -and (Get-FileHash -LiteralPath $versionPath -Algorithm SHA256).Hash -ne $expectedVersionHash) {
    throw 'CUDA version receipt changed. Do not resume an existing recipe or replace its recorded hash.'
}
Write-Host 'CUDA environment restored. Numerical acceptance and the original memory gate still apply before GPU training.'
