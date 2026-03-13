param(
    [int]$Port = 5006,
    [string]$Distro = "Ubuntu",
    [string]$RepoRootWindows = "",
    [string]$RepoRootWsl = "",
    [string]$ScriptRoot = "",
    [switch]$NoBrowser,
    [switch]$ForceNewServer
)

$ErrorActionPreference = "Stop"

function Test-ObservatoryReady {
    param(
        [int]$Port
    )

    try {
        $null = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Resolve-WslRepoPath {
    param(
        [string]$WindowsPath,
        [string]$Distro
    )

    $wslPath = & wsl.exe -d $Distro wslpath -a -u $WindowsPath 2>$null
    if (-not $wslPath) {
        throw "Unable to map '$WindowsPath' into WSL distro '$Distro'."
    }
    return ($wslPath | Out-String).Trim()
}

$effectiveScriptRoot = $ScriptRoot
if (-not $effectiveScriptRoot) {
    $effectiveScriptRoot = $PSScriptRoot
}

$configPath = Join-Path $effectiveScriptRoot "projection_observatory_launcher.json"
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
    if (-not $RepoRootWindows -and $config.RepoRootWindows) {
        $RepoRootWindows = $config.RepoRootWindows
    }
    if (-not $RepoRootWsl -and $config.RepoRootWsl) {
        $RepoRootWsl = $config.RepoRootWsl
    }
    if (
        $Distro -eq "Ubuntu" -and
        $config.Distro
    ) {
        $Distro = $config.Distro
    }
}

if (-not $RepoRootWindows) {
    $RepoRootWindows = (Resolve-Path (Join-Path $effectiveScriptRoot "..\..")).Path
}

if (-not $ForceNewServer -and (Test-ObservatoryReady -Port $Port)) {
    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$Port/" | Out-Null
    }
    exit 0
}

if (-not $RepoRootWsl) {
    $RepoRootWsl = Resolve-WslRepoPath -WindowsPath $RepoRootWindows -Distro $Distro
}

$wslCommand = "cd '$repoRootWsl' && source .venv/bin/activate && python scripts/analysis/observatory_dashboard.py --port $Port --no-open"
$cmdWindowCommand = 'title Projection Observatory && wsl.exe -d ' + $Distro + ' bash -lc "' + $wslCommand + '"'

Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList "/k", $cmdWindowCommand `
    -WorkingDirectory $RepoRootWindows | Out-Null

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if (Test-ObservatoryReady -Port $Port) {
        if (-not $NoBrowser) {
            Start-Process "http://127.0.0.1:$Port/" | Out-Null
        }
        exit 0
    }
}

Write-Warning "Projection Observatory did not answer on port $Port within 30 seconds. Check the WSL launcher window for startup errors."
