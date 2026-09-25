# Resumable upload to a rented GPU box (scp restarts from zero when a transfer stalls; sftp -a resumes).
# Usage: powershell -File Other\upload_resume.ps1 -Port 35556 -Host2 ssh5.vast.ai -Files a.npz,b.npz
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$Host2,
    [Parameter(Mandatory = $true)][string[]]$Files,
    [string]$Remote = '/workspace/sb',
    [int]$Tries = 40
)
$key = Join-Path (Split-Path -Parent $PSScriptRoot) 'Other\runpod_key'
foreach ($f in $Files) {
    $name = Split-Path $f -Leaf
    $size = (Get-Item -LiteralPath $f).Length
    for ($i = 1; $i -le $Tries; $i++) {
        $done = ssh -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $Port "root@$Host2" `
            "stat -c %s $Remote/$name 2>/dev/null || echo 0" 2>$null | Select-Object -Last 1
        if ([int64]$done -ge $size) { "$name complete ($([math]::Round($size/1MB)) MB)"; break }
        "$name : $([math]::Round([int64]$done/1MB)) / $([math]::Round($size/1MB)) MB - attempt $i"
        $batch = New-TemporaryFile
        "progress`n-put -a `"$($f -replace '\\', '/')`" $Remote/$name`nbye" | Set-Content $batch -Encoding ascii
        & sftp -a -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15 `
              -P $Port -b $batch "root@$Host2" *> $null
        Remove-Item $batch -Force -ErrorAction SilentlyContinue
    }
}
