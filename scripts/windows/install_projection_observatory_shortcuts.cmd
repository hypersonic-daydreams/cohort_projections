@echo off
setlocal
set "SCRIPT_PATH=%~dp0install_projection_observatory_shortcuts.ps1"
set "SCRIPT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$scriptPath = $env:SCRIPT_PATH; $scriptRoot = $env:SCRIPT_ROOT.TrimEnd('\'); $scriptContent = Get-Content -Raw $scriptPath; $scriptBlock = [scriptblock]::Create($scriptContent); & $scriptBlock -ScriptRoot $scriptRoot"
