param(
    [string]$ShortcutName = "Projection Observatory",
    [string]$ScriptRoot = ""
)

$ErrorActionPreference = "Stop"

$effectiveScriptRoot = $ScriptRoot
if (-not $effectiveScriptRoot) {
    $effectiveScriptRoot = $PSScriptRoot
}

$repoRootWindows = (Resolve-Path (Join-Path $effectiveScriptRoot "..\..")).ProviderPath
if ($repoRootWindows -notmatch '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.+)$') {
    throw "Expected the repository to be reachable from Windows via a WSL UNC path. Got: $repoRootWindows"
}

$distro = $matches[1]
$repoRootWsl = "/" + ($matches[2] -replace '\\', '/')

$installDir = Join-Path $env:LOCALAPPDATA "ProjectionObservatoryLauncher"
[System.IO.Directory]::CreateDirectory($installDir) | Out-Null

$launcherPs1 = Join-Path $installDir "Launch-ProjectionObservatory.ps1"
$launcherCmd = Join-Path $installDir "launch_projection_observatory.cmd"
$configPath = Join-Path $installDir "projection_observatory_launcher.json"

Copy-Item (Join-Path $effectiveScriptRoot "Launch-ProjectionObservatory.ps1") $launcherPs1 -Force
Copy-Item (Join-Path $effectiveScriptRoot "launch_projection_observatory.cmd") $launcherCmd -Force

$config = @{
    Distro = $distro
    RepoRootWindows = $repoRootWindows
    RepoRootWsl = $repoRootWsl
}
$config | ConvertTo-Json | Set-Content -Path $configPath -Encoding UTF8

$desktopPath = [Environment]::GetFolderPath("Desktop")
$startMenuPath = [Environment]::GetFolderPath("Programs")
$iconPath = Join-Path $env:SystemRoot "System32\wsl.exe"

$targets = @(
    @{
        LinkPath = Join-Path $desktopPath "$ShortcutName.lnk"
        Description = "Launch the Projection Observatory dashboard in WSL."
    },
    @{
        LinkPath = Join-Path $startMenuPath "$ShortcutName.lnk"
        Description = "Launch the Projection Observatory dashboard in WSL."
    }
)

$shell = New-Object -ComObject WScript.Shell

foreach ($target in $targets) {
    $shortcut = $shell.CreateShortcut($target.LinkPath)
    $shortcut.TargetPath = $launcherCmd
    $shortcut.WorkingDirectory = $installDir
    $shortcut.Description = $target.Description
    $shortcut.IconLocation = "$iconPath,0"
    $shortcut.Save()
    Write-Host "Created shortcut: $($target.LinkPath)"
}

Write-Host "Installed local launcher files in: $installDir"
