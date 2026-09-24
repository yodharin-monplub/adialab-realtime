# Daily backup of the SOURCE of this project to github.com/yodharin-monplub/adialab-realtime
# (AGENTS.md: "everyday, once per day when the cpu load is low, backup the project to github").
#
# Uses the repo deploy key Other/github_deploy_key (write access, this repo only) over SSH.
# What is backed up is decided by Other/gitignore.txt (whitelist: *.py, *.md, readme.txt, *.ps1 ...);
# data, caches, model dumps, submission bundles and every credential file are excluded.
# It waits for the CPU to be quiet, so it never competes with a training run.
#   manual run:  powershell -ExecutionPolicy Bypass -File Other\backup_github.ps1
#   scheduled:   task "ClaudeStructuralBreakBackup", daily 04:30 (schtasks /query /tn ...)
param([int]$MaxWaitMinutes = 120, [int]$BusyPercent = 60)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')

# wait for a quiet CPU (training has priority)
$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
while ((Get-Date) -lt $deadline) {
    $load = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples.CookedValue |
            Measure-Object -Average | ForEach-Object Average
    if ($load -lt $BusyPercent) { break }
    Start-Sleep -Seconds 300
}

git add -A
$changes = git status --porcelain | Measure-Object | ForEach-Object Count
if ($changes -eq 0) { "$(Get-Date -Format s)  nothing changed, no commit"; exit 0 }

$message = "Daily backup $(Get-Date -Format 'yyyy-MM-dd')`n`nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git commit -q -m $message
git push -q origin main
"$(Get-Date -Format s)  pushed $changes changed files"
