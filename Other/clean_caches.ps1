# Clean regenerable caches of this project.  Safe: Python rebuilds __pycache__ automatically,
# pip re-downloads packages only if a package is installed again.
#   Report only (default):   powershell -ExecutionPolicy Bypass -File Other\clean_caches.ps1
#   Actually delete:         powershell -ExecutionPolicy Bypass -File Other\clean_caches.ps1 -Apply
# Run it from the project root.  It does NOT touch data, models, submissions or the .venv packages,
# and it skips the hidden link folders (models, train, submissions, .venv) so nothing is visited twice.
param([switch]$Apply)

$root = Split-Path -Parent $PSScriptRoot
$skip = @('models', 'train', 'submissions', '.venv')

function Get-SizeGB($path) {
    $s = (Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $s) { 0 } else { [math]::Round($s / 1GB, 3) }
}

# 1. __pycache__ folders under the real (non-link) folders of the project root
$targets = @()
Get-ChildItem $root -Force -Directory | Where-Object { $skip -notcontains $_.Name } | ForEach-Object {
    if ($_.Name -eq '__pycache__') { $targets += $_ }
    else { $targets += Get-ChildItem $_.FullName -Recurse -Force -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -notlike '*\Other\.venv\*' } }
}
$total = ($targets | ForEach-Object { Get-SizeGB $_.FullName } | Measure-Object -Sum).Sum
"__pycache__ folders: $($targets.Count), $total GB"

# 2. pip download cache (outside the project)
$pip = Join-Path $env:LOCALAPPDATA 'pip\cache'
"pip cache ($pip): $(Get-SizeGB $pip) GB"

if (-not $Apply) { "Report only. Re-run with -Apply to delete the above."; return }

$targets | ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path $pip) { Remove-Item (Join-Path $pip '*') -Recurse -Force -ErrorAction SilentlyContinue }
"Done. Free on C: now {0:N2} GB" -f ((Get-PSDrive C).Free / 1GB)
