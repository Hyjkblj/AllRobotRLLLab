$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\conda\envs\allrobotrl-platform\python.exe"
$npmExe = "D:\Develop\nodejs\npm.cmd"
$serviceScript = Join-Path $PSScriptRoot "mujoco_service.py"
$reactRoot = Join-Path $PSScriptRoot "react-app"

if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Conda environment executable not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $serviceScript)) {
  throw "MuJoCo service not found: $serviceScript"
}
if (-not (Test-Path -LiteralPath $npmExe)) {
  throw "Node/npm executable not found: $npmExe"
}

$listeners = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
  $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
  if ($process -and $process.Path -eq $pythonExe) {
    Stop-Process -Id $process.Id -Force
  }
}

Start-Process -FilePath $pythonExe -ArgumentList $serviceScript -WorkingDirectory $projectRoot -WindowStyle Hidden
Start-Process -FilePath $npmExe -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") -WorkingDirectory $reactRoot

Write-Output "MuJoCo API: http://127.0.0.1:8787"
Write-Output "React UI:   http://127.0.0.1:4173"
