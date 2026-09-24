# Layout required by AGENTS.md (2026-09-20):
#   <root>\AGENTS.md   Data\   Model\   Other\        (nothing else visible at the root)
#
# WHY THERE ARE HIDDEN LINKS AT THE ROOT
#   The frozen training pipeline (Model20 ... Model84 protocol.json files) verifies the SHA-256 of
#   source files BY THEIR ORIGINAL RELATIVE PATH ("models/model84/run_model84.py", "tsauc.py", ...)
#   and those protocol files are hash-pinned themselves, so neither the code nor the paths may change.
#   Therefore the real folders are moved/renamed and hidden compatibility links keep the old paths alive:
#       models       -> Model                                   (junction, hidden)
#       train        -> Data\ADIA Lab Structural Break Challenge Real-Time Edition   (junction, hidden)
#       submissions  = the REAL folder, hidden; Other\submissions is a junction to it (a path check in
#                      models/model20/features.py resolves links, so this one cannot be the link)
#       .venv        -> Other\.venv                             (junction, hidden)
#       tsauc.py, .env, email.txt                               (hidden copies of the files in Other\)
#   No source file is edited.  Run  `Other\restructure_layout.ps1 -LinksOnly`  on another PC (or after
#   OneDrive restored the folder) to recreate the links.
#
# Usage:  powershell -ExecutionPolicy Bypass -File Other\restructure_layout.ps1 [-DryRun] [-LinksOnly]
param([switch]$DryRun, [switch]$LinksOnly)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $PSScriptRoot "restructure_log.txt"
$rt = "Data\ADIA Lab Structural Break Challenge Real-Time Edition"

function Step($text, [scriptblock]$action) {
    Add-Content -Path $log -Value ("{0}  {1}{2}" -f (Get-Date -Format s), $(if ($DryRun) { "[dry-run] " } else { "" }), $text)
    Write-Output $text
    if (-not $DryRun) { & $action }
}

function Link($name, $target) {
    if (Test-Path $name) { return }
    Step "junction  $name -> $target" {
        cmd /c mklink /J "$name" "$target" | Out-Null
        cmd /c attrib +h +s "$name" /L | Out-Null
    }
}

function HiddenCopy($name) {
    if (-not (Test-Path "Other\$name") -or (Test-Path $name)) { return }   # never overwrites or deletes anything
    Step "hidden copy  $name  (from Other\$name)" {
        Copy-Item "Other\$name" $name
        attrib +h $name
    }
}

if (-not $LinksOnly) {
    if (-not $DryRun -and (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 200MB })) {
        throw "python jobs are still running - stop them first"
    }
    if ((Test-Path "data") -and -not (Test-Path "Data_tmp")) {
        Step "rename  data -> Data" { Rename-Item "data" "Data_tmp"; Rename-Item "Data_tmp" "Data" }
    }
    if ((Test-Path "train") -and ((Get-Item "train").LinkType -ne "Junction")) {
        foreach ($f in Get-ChildItem "train" -File) {
            $same = (Get-FileHash $f.FullName).Hash -eq (Get-FileHash (Join-Path $rt $f.Name)).Hash
            if (-not $same) { throw "train\$($f.Name) differs from the copy in $rt" }
        }
        Step "move  train -> Other\_duplicates\train  (byte-identical to $rt; safe for the user to delete)" {
            New-Item -ItemType Directory -Force "Other\_duplicates" | Out-Null
            Move-Item "train" "Other\_duplicates\train"
        }
    }
    if ((Test-Path "models") -and ((Get-Item "models").LinkType -ne "Junction")) {
        Step "rename  models -> Model" { Rename-Item "models" "Model" }
    }
    foreach ($name in "tests", ".venv") {
        if ((Test-Path $name) -and ((Get-Item $name -Force).LinkType -ne "Junction")) {
            Step "move  $name -> Other\$name" { Move-Item $name "Other\$name" }
        }
    }
    foreach ($name in ".env", "email.txt", "tsauc.py", "README.md", "requirements.txt", "discord.txt", "overlap_result.txt", ".gitignore") {
        if ((Test-Path $name) -and -not (Test-Path "Other\$name")) {
            Step "move  $name -> Other\$name" { Move-Item $name "Other\$name" }
        }
    }
}

Link "models" "Model"
Link "train" $rt
if (Test-Path "submissions") { cmd /c attrib +h +s submissions | Out-Null }   # REAL folder: model20/features.py compares the resolved detector path with <root>\submissions\submission16
Link "Other\submissions" "submissions"
if (Test-Path "__pycache__") { cmd /c attrib +h __pycache__ | Out-Null }
Link ".venv" "Other\.venv"
foreach ($name in "tsauc.py", ".env", "email.txt") { HiddenCopy $name }

Write-Output "`nvisible at the root:"
Get-ChildItem | Select-Object -ExpandProperty Name
Write-Output "`nhidden compatibility links at the root:"
Get-ChildItem -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::Hidden } | Select-Object -ExpandProperty Name
