param(
    [Parameter(Mandatory = $true)][string]$InputAudio,
    [ValidateRange(1, 10)][int]$Repeats = 3,
    [string]$OutputDirectory = 'output/arc-benchmark'
)
$ErrorActionPreference = 'Stop'
$audio = (Resolve-Path -LiteralPath $InputAudio).Path
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$records = @()

# One separate GPU warm-up populates the compilation cache before measured runs.
$warm = Join-Path $OutputDirectory 'gpu-warmup.txt'
& ./target/release/autotab.exe --input $audio --device openvino -o $warm
if ($LASTEXITCODE -ne 0) { throw 'Intel GPU warm-up failed' }

for ($iteration = 1; $iteration -le $Repeats; $iteration++) {
    foreach ($device in @('cpu', 'openvino')) {
        $prefix = Join-Path $OutputDirectory "$device-$iteration"
        & ./target/release/autotab.exe --input $audio --device $device --precision fp32 -o "$prefix.txt" --timings-json "$prefix.json" --profile-dir "$prefix-profile" --guitar-output "$prefix.wav"
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $device, run $iteration" }
        $record = Get-Content "$prefix.json" -Raw | ConvertFrom-Json
        $record | Add-Member -NotePropertyName iteration -NotePropertyValue $iteration
        $records += $record
        if ($device -eq 'openvino') {
            $profile = Get-Content "$prefix-profile/demucs-openvino.json" -Raw | ConvertFrom-Json
            if ($profile.execution_devices -notmatch 'GPU' -or $profile.inference_precision -ne 'f32') {
                throw 'Benchmark did not execute on the GPU in FP32'
            }
            if (-not $profile.loaded_from_cache) { throw 'GPU model cache was not reused across processes' }
        }
    }
}

function Median($Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count % 2) { return $sorted[[int][math]::Floor($sorted.Count / 2)] }
    return ($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2
}
$cpu = Median @($records | Where-Object device -eq 'Cpu' | ForEach-Object total_seconds)
$gpu = Median @($records | Where-Object device -eq 'Openvino' | ForEach-Object total_seconds)
$report = [ordered]@{
    audio = $audio
    hardware = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    repetitions = $Repeats
    cpu_median_seconds = $cpu
    gpu_median_seconds = $gpu
    speedup = $cpu / $gpu
    measurements = $records
}
$report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host ('CPU {0:N2}s; Arc {1:N2}s; speedup {2:N2}x (warm cache, FP32)' -f $cpu, $gpu, ($cpu / $gpu))
